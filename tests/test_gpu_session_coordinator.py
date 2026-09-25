from app.services.gpu_session_coordinator import GpuSessionCoordinator


class FakeOwner:
    def __init__(self, name: str) -> None:
        self.name = name
        self.released: list[str] = []

    def release_device(self, device: str) -> None:
        self.released.append(device)


def test_acquire_new_device_does_not_release_anything():
    coordinator = GpuSessionCoordinator()
    owner = FakeOwner("a")
    coordinator.acquire("dml:0", owner)
    assert owner.released == []


def test_acquire_same_device_different_owner_releases_previous():
    coordinator = GpuSessionCoordinator()
    owner_a = FakeOwner("a")
    owner_b = FakeOwner("b")
    coordinator.acquire("dml:0", owner_a)
    coordinator.acquire("dml:0", owner_b)
    assert owner_a.released == ["dml:0"]
    assert owner_b.released == []


def test_acquire_same_device_same_owner_does_not_release():
    coordinator = GpuSessionCoordinator()
    owner = FakeOwner("a")
    coordinator.acquire("dml:0", owner)
    coordinator.acquire("dml:0", owner)
    assert owner.released == []


def test_acquire_different_devices_never_release_each_other():
    coordinator = GpuSessionCoordinator()
    owner_a = FakeOwner("a")
    owner_b = FakeOwner("b")
    coordinator.acquire("dml:0", owner_a)
    coordinator.acquire("dml:1", owner_b)
    assert owner_a.released == []
    assert owner_b.released == []


def test_acquire_cpu_device_tracked_independently_from_gpu_devices():
    coordinator = GpuSessionCoordinator()
    owner_cpu = FakeOwner("cpu-user")
    owner_gpu = FakeOwner("gpu-user")
    coordinator.acquire("cpu", owner_cpu)
    coordinator.acquire("dml:0", owner_gpu)
    assert owner_cpu.released == []
    assert owner_gpu.released == []


def test_owner_regains_device_after_being_released():
    coordinator = GpuSessionCoordinator()
    owner_a = FakeOwner("a")
    owner_b = FakeOwner("b")
    coordinator.acquire("dml:0", owner_a)
    coordinator.acquire("dml:0", owner_b)
    coordinator.acquire("dml:0", owner_a)
    assert owner_a.released == ["dml:0"]  # fue liberado cuando owner_b tomo el device
    assert owner_b.released == ["dml:0"]  # fue liberado cuando owner_a lo retomo


class RaisingOwner(FakeOwner):
    def release_device(self, device: str) -> None:
        super().release_device(device)
        raise RuntimeError("the session was already dead")


def test_invalidate_device_releases_that_device_on_every_owner_that_ever_acquired():
    coordinator = GpuSessionCoordinator()
    owner_a = FakeOwner("a")
    owner_b = FakeOwner("b")
    owner_c = FakeOwner("c")
    coordinator.acquire("dml:0", owner_a)
    coordinator.acquire("dml:0", owner_b)
    coordinator.acquire("dml:1", owner_c)

    coordinator.invalidate_device("dml:0")

    assert owner_a.released == ["dml:0", "dml:0"]
    assert owner_b.released == ["dml:0"]
    assert owner_c.released == ["dml:0"]


def test_invalidate_device_reaches_registered_owners_that_never_acquired():
    coordinator = GpuSessionCoordinator()
    owner = FakeOwner("idle")
    coordinator.register(owner)

    coordinator.invalidate_device("dml:0")

    assert owner.released == ["dml:0"]


def test_register_is_idempotent_so_each_owner_is_released_once():
    coordinator = GpuSessionCoordinator()
    owner = FakeOwner("a")
    coordinator.register(owner)
    coordinator.register(owner)
    coordinator.acquire("dml:0", owner)

    coordinator.invalidate_device("dml:0")

    assert owner.released == ["dml:0"]


def test_invalidate_device_leaves_other_devices_untouched():
    coordinator = GpuSessionCoordinator()
    owner_a = FakeOwner("a")
    owner_b = FakeOwner("b")
    coordinator.acquire("dml:0", owner_a)
    coordinator.acquire("dml:1", owner_b)

    coordinator.invalidate_device("dml:0")
    coordinator.acquire("dml:1", owner_a)

    assert owner_a.released == ["dml:0"]
    assert owner_b.released == ["dml:0", "dml:1"]


def test_after_invalidation_nobody_owns_the_device():
    coordinator = GpuSessionCoordinator()
    owner_a = FakeOwner("a")
    owner_b = FakeOwner("b")
    coordinator.acquire("dml:0", owner_a)
    coordinator.invalidate_device("dml:0")

    coordinator.acquire("dml:0", owner_b)

    assert owner_a.released == ["dml:0"]
    assert owner_b.released == []


def test_an_owner_failing_to_release_does_not_stop_the_broadcast(caplog):
    coordinator = GpuSessionCoordinator()
    broken = RaisingOwner("broken")
    healthy = FakeOwner("healthy")
    coordinator.register(broken)
    coordinator.register(healthy)

    coordinator.invalidate_device("dml:0")

    assert broken.released == ["dml:0"]
    assert healthy.released == ["dml:0"]
    assert "the session was already dead" in caplog.text
