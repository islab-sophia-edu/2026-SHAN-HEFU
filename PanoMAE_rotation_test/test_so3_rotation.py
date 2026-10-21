import torch
import torch.nn.functional as F
from so3_rotation import ERPSO3Rotator, rotation_matrix


def test_identity():
    x = torch.rand(2, 3, 24, 48)
    assert torch.equal(ERPSO3Rotator().rotate(x), x)


def test_yaw_full_period():
    x = torch.rand(1, 3, 24, 48)
    y = ERPSO3Rotator().rotate(x, yaw=360)
    assert torch.allclose(x, y, atol=2e-5)


def test_constant_world_normal_rotation():
    x = F.normalize(torch.tensor([1., 2., 3.]), dim=0).reshape(1, 3, 1, 1).expand(1, 3, 24, 48)
    y = ERPSO3Rotator().rotate(x, yaw=45, pitch=20, roll=-10, vector=True)
    R = rotation_matrix(45, 20, -10, device=x.device)
    expected = (R @ x[0, :, 0, 0]).reshape(1, 3, 1, 1)
    assert torch.allclose(y, expected.expand_as(y), atol=2e-5)


if __name__ == '__main__':
    test_identity()
    test_yaw_full_period()
    test_constant_world_normal_rotation()
    print('All tests passed')
