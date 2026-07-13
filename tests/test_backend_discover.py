from omnirun.backends.base import Backend, register
from omnirun.config import BackendConfig, GpuDecl
from omnirun.models import (
    CancelMode,
    Health,
    JobHandle,
    Offer,
    ResourceSpec,
    StatusReport,
)


@register("discotest")
class _DiscoBackend(Backend):
    def probe(self, res: ResourceSpec) -> list[Offer]:
        return []

    def submit(self, spec, offer, on_provisioning=None) -> JobHandle:
        raise NotImplementedError

    def status(self, handle) -> StatusReport:
        raise NotImplementedError

    def logs(self, handle, follow: bool = False):
        raise NotImplementedError

    def cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None:
        raise NotImplementedError

    def pull_outputs(self, handle, dest):
        raise NotImplementedError

    def check(self) -> str:
        if self.config.extra("broken"):
            raise RuntimeError("cannot reach box")
        return "ok"


def test_default_discover_uses_declared_gpus_and_ok_health():
    cfg = BackendConfig(type="discotest", gpus=[GpuDecl(type="A100-80", count=4)])
    facts = _DiscoBackend("box", cfg).discover()
    assert facts.backend == "box"
    assert facts.capabilities.gpu_types == ["A100-80"]
    assert facts.health == Health.OK


def test_default_discover_marks_unreachable_on_check_failure():
    cfg = BackendConfig.model_validate({"type": "discotest", "broken": True})
    facts = _DiscoBackend("box", cfg).discover()
    assert facts.health == Health.UNREACHABLE
    assert "cannot reach box" in facts.health_detail
