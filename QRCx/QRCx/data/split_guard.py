"""Test-split (2024) access guard — Sprint 3 closeout addendum.

Canonical split: train 2019-2022 / val 2023 / test 2024. All development
decisions — hyperparameter tuning, model selection, DoD/gate checks — must
be made on val only. Test-2024 is reserved for a final, one-time
confirmatory report, not for iterative sprint-by-sprint development.

Any code that scores predictions against a test-2024 split (`X_test`/
`y_test`/`test_seq` from `data.preprocessor.preprocess()`) must first call
`assert_test_unlocked()`. It raises `TestSplitLockedError` unless the
caller passes the exact unlock token (reserved for the final Sprint 6
confirmatory runner) or the `QRCX_UNLOCK_TEST_SPLIT=1` environment
variable is set (for deliberately reproducing a historical, already-
logged test-touch entry — see docs/evaluation_protocol.md's "test-set
touch ledger").
"""
import os

UNLOCK_TOKEN = "sprint6_runner"
_UNLOCK_ENV = "QRCX_UNLOCK_TEST_SPLIT"


class TestSplitLockedError(RuntimeError):
    pass


def assert_test_unlocked(unlocked_by: str = "", reason: str = "") -> None:
    """Raise TestSplitLockedError unless explicitly unlocked.

    Args:
        unlocked_by: pass `split_guard.UNLOCK_TOKEN` — intended only for
            the final Sprint 6 confirmatory runner, not iterative
            development.
        reason: free-text context appended to the error message, to make
            an accidental/blocked call easy to trace back to its caller.
    """
    if unlocked_by == UNLOCK_TOKEN:
        return
    if os.environ.get(_UNLOCK_ENV) == "1":
        return
    msg = (
        "Test split (2024) is locked for development use per the Sprint 3 "
        "closeout addendum — all decisions and DoD/gate checks must be "
        "made on val-2023. Pass unlocked_by=split_guard.UNLOCK_TOKEN (final "
        "Sprint 6 confirmatory run only) or set QRCX_UNLOCK_TEST_SPLIT=1 "
        "to deliberately reproduce a logged historical test-touch entry "
        "(see docs/evaluation_protocol.md)."
    )
    if reason:
        msg += f" Context: {reason}"
    raise TestSplitLockedError(msg)
