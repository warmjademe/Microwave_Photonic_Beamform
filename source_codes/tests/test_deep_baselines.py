"""合成数据上的接口、复乘、融合、划分与量化验证；仅在华硕运行。"""
import unittest
import numpy as np
import torch
from deep_common.layers import build, METHODS, split_input
from deep_common.preprocessing import fit, apply
from baseline_complex_cnn.method import ComplexConv
from baseline_jct.method import JCTBlock


class Tests(unittest.TestCase):
    def test_all_models_gradient_and_shape(self):
        torch.set_num_threads(1)
        x=torch.randn(3,2513)
        for method in METHODS:
            model=build(method);out=model(x)
            self.assertEqual(tuple(out.shape),(3,128))
            self.assertTrue(bool(((out>=0)&(out<=1)).all()))
            out.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_iq_token_identity(self):
        raw=torch.arange(2513.).reshape(1,-1)
        tokens,aux=split_input(raw)
        for probe,frequency,pilot in [(0,0,0),(15,30,1),(7,4,0)]:
            index=((probe*31+frequency)*2+pilot)*2
            self.assertTrue(torch.equal(tokens[0,frequency,(probe*2+pilot)*2:(probe*2+pilot)*2+2],raw[0,index:index+2]))

    def test_cuda_training_step(self):
        self.assertTrue(torch.cuda.is_available(), '实验主机必须提供CUDA')
        torch.use_deterministic_algorithms(True)
        for method in METHODS:
            model=build(method).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            loss=(model(torch.randn(8,2513,device='cuda'))-torch.rand(8,128,device='cuda')).square().mean()
            optimizer.zero_grad();loss.backward();optimizer.step();torch.cuda.synchronize()
            self.assertTrue(torch.isfinite(loss).item())

    def test_complex_multiplication(self):
        layer=ComplexConv(1,1)
        with torch.no_grad():
            layer.a.weight.zero_();layer.b.weight.zero_()
            layer.a.weight[0,0,1]=2;layer.b.weight[0,0,1]=3
        r=torch.tensor([[[1.,2.,3.]]]);i=torch.tensor([[[4.,5.,6.]]])
        real,imag=layer((r,i))
        torch.testing.assert_close(real,2*r-3*i);torch.testing.assert_close(imag,3*r+2*i)

    def test_jct_fusion(self):
        layer=JCTBlock();x=torch.randn(2,31,128)
        with torch.no_grad():
            weights=layer.gate(x.mean(1))
            torch.testing.assert_close(weights.sum(-1),torch.ones(2))
            expected=weights[:,0,None,None]*layer.cnn(x.transpose(1,2)).transpose(1,2)+weights[:,1,None,None]*layer.transformer(x)
            torch.testing.assert_close(layer(x),expected)

    def test_training_only_scaling(self):
        class Data:
            split='test'
            X=np.ones((8,2513),np.float32)
            def __len__(self): return len(self.X)
        d=Data()
        with self.assertRaises(ValueError):fit(d)
        d.split='train';d.X[:,0]=np.arange(8);d.X[:,1]=2*np.arange(8)
        stats=fit(d);self.assertEqual(stats['scale'][0],stats['scale'][1])
        self.assertTrue(np.isfinite(apply(d.X,stats)).all())

    def test_hardware_quantization(self):
        from native_sim.config import NativeConfig
        from baseline_common.controls import codes
        levels=np.r_[np.full(64,76),np.full(64,24)]
        u=np.r_[np.full(64,2.5/76),np.full(64,3.5/24)]
        result=codes(u,NativeConfig())
        np.testing.assert_array_equal(result,np.r_[np.full(64,3),np.full(64,4)])
        self.assertTrue(np.all(result<=levels))

    def test_nearly_constant_noise_variance(self):
        class Data:
            split='train'
            X=np.ones((10000,2513),np.float32)
            def __len__(self):return len(self.X)
        d=Data();d.X[:,2001]=1e-8*(1+np.linspace(-1e-6,1e-6,len(d)))
        stats=fit(d);z=apply(d.X,stats)
        self.assertLess(float(abs(z).max()),5.)
        self.assertAlmostEqual(float(z[:,2001].std()),1.,places=4)


if __name__=='__main__':unittest.main()
