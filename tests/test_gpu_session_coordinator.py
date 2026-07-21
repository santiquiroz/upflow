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
