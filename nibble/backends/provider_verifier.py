"""
Provider Verification and Execution Tracer for Nibble.
Ensures transparent logging of requested vs actual execution providers,
traps silent fallbacks (e.g. GPU/QNN silently falling back to CPU),
and formats execution traces for auditability.
"""

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
from nibble.utils.hashing import calculate_file_sha256


@dataclass
class ExecutionTrace:
    model_name: str
    model_sha256: str
    requested_provider: str
    actual_provider: str
    fallback_occurred: bool
    fallback_reason: Optional[str] = None
    node_count: int = 0
    device_name: str = "Host Processor"
    status: str = "SUCCESS"

    def format_summary(self) -> str:
        fallback_str = f"Yes ({self.fallback_reason})" if self.fallback_occurred else "No"
        return (
            f"Execution Trace:\n"
            f"  Model:              {self.model_name}\n"
            f"  Model SHA-256:      {self.model_sha256[:16]}...\n"
            f"  Requested Provider: {self.requested_provider}\n"
            f"  Actual Provider:    {self.actual_provider}\n"
            f"  Nodes:              {self.node_count}\n"
            f"  Fallback Occurred:  {fallback_str}\n"
            f"  Status:             {self.status}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProviderVerifier:
    """Verifies that an ONNX Runtime InferenceSession used the intended provider."""

    @staticmethod
    def verify(
        session,
        requested_provider: str,
        model_path: str | Path,
        device_name: str = ""
    ) -> ExecutionTrace:
        p = Path(model_path)
        model_hash = calculate_file_sha256(p)
        model_name = p.name

        active_providers = session.get_providers() if hasattr(session, "get_providers") else []
        actual_provider = active_providers[0] if active_providers else "Unknown"

        # Determine if fallback occurred
        fallback = False
        reason = None

        req_clean = requested_provider.strip().lower()
        act_clean = actual_provider.strip().lower()

        if "qnn" in req_clean and "qnn" not in act_clean:
            fallback = True
            reason = f"QNNExecutionProvider was requested but session executed on {actual_provider}"
        elif ("dml" in req_clean or "directml" in req_clean) and "dml" not in act_clean:
            fallback = True
            reason = f"DmlExecutionProvider was requested but session executed on {actual_provider}"

        node_count = 0
        try:
            import onnx
            m = onnx.load(str(p))
            node_count = len(m.graph.node)
        except Exception:
            pass

        return ExecutionTrace(
            model_name=model_name,
            model_sha256=model_hash,
            requested_provider=requested_provider,
            actual_provider=actual_provider,
            fallback_occurred=fallback,
            fallback_reason=reason,
            node_count=node_count,
            device_name=device_name or actual_provider,
            status="WARNING_FALLBACK" if fallback else "SUCCESS"
        )
