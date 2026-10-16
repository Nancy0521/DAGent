"""Bounded, stage-local repair. Never multiply transport retries invisibly."""
from dagent.core.api import APIError, safe_error_metadata
from dagent.core.budget import BudgetExceeded

ATTEMPT_LIMIT = 3
PERMANENT = {'authentication', 'payment_or_balance', 'permission', 'endpoint_or_model',
             'quota_or_balance', 'invalid_request', 'request_too_large'}


def failure_feedback(exc, api=None):
    diagnostic = safe_error_metadata(exc)
    # APIClient wraps provider errors; the settled ledger retains safe categories.
    events = getattr(getattr(api, 'ledger', None), 'events', [])
    if isinstance(exc, APIError) and events and events[-1].get('status') == 'error':
        diagnostic = {k: events[-1].get(k, v) for k, v in diagnostic.items()}
    permanent = (isinstance(exc, BudgetExceeded) or diagnostic.get('error_category') in PERMANENT
                 or (isinstance(exc, APIError) and str(exc).startswith('Missing API credential')))
    return dict(error_type=type(exc).__name__, diagnostic=diagnostic, retryable=not permanent,
                validation_error=str(exc)[:1600] if isinstance(exc, ValueError) else None)


class FeedbackAPI:
    """Attach prior failed validation to a NEW request, not a cached replay."""
    def __init__(self, api, attempt, feedback):
        self.api, self.attempt, self.feedback = api, attempt, feedback

    def __getattr__(self, name):
        return getattr(self.api, name)

    def json(self, stage, system, prompt, **kwargs):
        import json
        if self.attempt:
            prompt += ('\nREPAIR ATTEMPT ' + str(self.attempt + 1) + ' OF 3. '
                       'The prior attempt failed the checks below. Correct these errors using only '
                       'the supplied source blocks; return the complete requested JSON. '
                       'Do not invent evidence or change locked metadata.\n'
                       + json.dumps(self.feedback, ensure_ascii=False))
            stage += f'_repair_{self.attempt}'
        return self.api.json(stage, system, prompt, **kwargs)
