"""Offline regression tests for the final repository-local benchmark default."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import fitz

from dagent.core.config import Config, ROOT, dump_json
from dagent.core.document import sha256
from dagent.agents import dataset, runner


class LocalDatasetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=ROOT / 'outputs')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.benchmark = self.root / 'benchmark'
        self.papers = self.benchmark / 'papers'
        self.papers.mkdir(parents=True)
        guard = patch.object(runner.APIClient, 'json', side_effect=AssertionError('Real API forbidden'))
        guard.start()
        self.addCleanup(guard.stop)

    def pdf(self, name='001_Final source.pdf'):
        path = self.papers / name
        with fitz.open() as doc:
            page = doc.new_page()
            page.insert_text((50, 50), 'Final source paper')
            doc.save(path)
        return path

    def test_inventory_reads_only_source_pdfs_and_keeps_all_files(self):
        pdf = self.pdf()
        shutil.copy2(pdf, self.papers / '002_Duplicate.PDF')
        (self.papers / '003_Invalid.pdf').write_bytes(b'not a PDF')
        (self.papers / 'notes.txt').write_text('not a source')
        for folder in ('meta', 'posters'):
            (self.benchmark / folder).mkdir()
            (self.benchmark / folder / 'manifest.json').write_text('NOT JSON; do not read')
        before = {str(p): sha256(p) for p in self.benchmark.rglob('*') if p.is_file()}
        manifest = dataset.discover_sources(self.benchmark)
        self.assertEqual(manifest['root'], str(self.benchmark))
        self.assertEqual(manifest['paper_count'], 3)
        self.assertEqual(manifest['ready_count'], 2)
        self.assertEqual(manifest['invalid_input_count'], 1)
        self.assertEqual(manifest['duplicate_pdf_groups'], [['001_Final_source', '002_Duplicate']])
        self.assertFalse(manifest['benchmark_meta_or_reference_posters_read'])
        self.assertFalse(manifest['ground_truth_used_for_generation'])
        self.assertFalse(manifest['parsed_cache_used'])
        for entry in manifest['papers']:
            self.assertTrue(Path(entry['pdf']).is_relative_to(self.papers))
            self.assertEqual(entry['pdf_sha256'], sha256(Path(entry['pdf'])))
            for field in ('parsed_json', 'ground_truth', 'qa_path', 'source_metadata'):
                self.assertIsNone(entry[field])
        after = {str(p): sha256(p) for p in self.benchmark.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_empty_or_missing_directory_fails_without_old_dataset_fallback(self):
        with self.assertRaisesRegex(ValueError, 'No benchmark PDFs'):
            dataset.discover_sources(self.benchmark)
        with self.assertRaisesRegex(FileNotFoundError, 'papers directory not found'):
            dataset.discover_sources(self.root / 'absent')

    def test_pdf_symlink_outside_dataset_is_rejected(self):
        outside = self.root / 'old.pdf'
        outside.write_bytes(b'old dataset')
        (self.papers / '001_Outside.pdf').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'outside papers directory'):
            dataset.discover_sources(self.benchmark)

    def test_papers_directory_symlink_outside_dataset_is_rejected(self):
        other = self.root / 'other'
        other.mkdir()
        (other / 'papers').symlink_to(self.papers, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'outside the dataset'):
            dataset.discover_sources(other)

    def test_colliding_source_ids_are_rejected(self):
        first = self.pdf('001_A B.pdf')
        shutil.copy2(first, self.papers / '001_A_B.pdf')
        with self.assertRaisesRegex(ValueError, 'Duplicate benchmark paper ID'):
            dataset.discover_sources(self.benchmark)

    def config(self):
        config = self.root / 'config.json'
        dump_json(config, Config().to_dict())
        return config

    def test_prepare_freezes_fresh_local_manifest_and_its_hash(self):
        self.pdf()
        run = self.root / 'run'
        with patch.object(runner, 'sources', return_value=[Path(dataset.__file__)]):
            runner.prepare(run, config_path=self.config(), benchmark_root=self.benchmark)
        manifest = json.loads((run / 'benchmark_manifest.json').read_text())
        protocol = json.loads((run / 'protocol.json').read_text())
        self.assertEqual(manifest['paper_count'], 1)
        self.assertEqual(manifest['root'], str(self.benchmark))
        self.assertEqual(protocol['manifest_sha256'], sha256(run / 'benchmark_manifest.json'))
        self.assertTrue(protocol['full_benchmark'])
        self.assertEqual(protocol['entries'][0]['pdf'], str(self.papers / '001_Final source.pdf'))
        self.assertEqual(protocol['entries'][0]['input_recovery']['status'], 'original')
        self.assertFalse((self.benchmark / 'benchmark_manifest.json').exists())
        runner.frozen(run)  # Newly frozen source/config/manifest hashes remain valid.

    def test_explicit_manifest_override_is_preserved(self):
        self.pdf()
        manifest = self.root / 'explicit.json'
        dump_json(manifest, dataset.discover_sources(self.benchmark))
        run = self.root / 'explicit-run'
        with (patch.object(runner, 'sources', return_value=[]),
              patch.object(runner, 'discover_sources', side_effect=AssertionError('Override ignored'))):
            runner.prepare(run, manifest, self.config())
        self.assertEqual(manifest.read_bytes(), (run / 'benchmark_manifest.json').read_bytes())

    def test_runner_cli_defaults_to_repository_benchmark_not_old_manifest(self):
        self.assertIsNone(runner.DEFAULT_MANIFEST)
        self.assertEqual(dataset.DEFAULT_BENCHMARK_ROOT, ROOT / 'benchmark')
        with patch.object(runner, 'prepare', return_value={}) as prepare, patch('builtins.print'):
            runner.main(['prepare', '--run-dir', str(self.root / 'run')])
        self.assertIsNone(prepare.call_args.args[1])
        self.assertEqual(prepare.call_args.kwargs['benchmark_root'], str(ROOT / 'benchmark'))


if __name__ == '__main__':
    unittest.main()
