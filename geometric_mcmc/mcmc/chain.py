# Author: Lianghao Cao
# Date: 04/05/2023
import hippylib as hp
import numpy as np
import time
from .kernel import Kernel, DelayedAcceptanceKernel
from .tracer import Tracer, NullTracer
from typing import Any

class MCMC:
    """
    The MCMC class that handles global (beyond the Metropolis--Hasting step) MCMC operations
    """
    def __init__(self, kernel: Kernel) -> None:
        """
        :param kernel: The kernel object
        """
        self.kernel = kernel
        self.parameters = {}
        self.parameters["number_of_samples"] = None
        self.parameters["print_progress"] = 20
        self.parameters["print_level"] = 1

    def run(self, m0: Any, qoi=None, tracer: Tracer=None) -> int:
        if qoi is None:
            qoi = hp.NullQoi()
        if tracer is None:
            tracer = NullTracer()
        
        assert self.parameters["number_of_samples"] is not None, "The number of samples is not set"
        number_of_samples = self.parameters["number_of_samples"]

        current = self.kernel.generate_sample()
        proposed = self.kernel.generate_sample()

        current.assign_paramaeter(m0)

        self.kernel.init_sample(current)

        if self.parameters["print_level"] > 0:
            print("Generate {0} samples".format(number_of_samples))
        sample_count = 0
        naccept = 0
        n_check = number_of_samples // self.parameters["print_progress"]
        t0 = time.time()
        while (sample_count < number_of_samples):
            self.kernel.sample(current, proposed)
            self.kernel.init_sample(proposed)
            accept = self.kernel.accept_or_reject(current, proposed)
            if accept == 1:
                current.assign(proposed)
            naccept += accept
            q = qoi.eval([current.u, current.m])
            tracer.append(current, q, accept)
            sample_count += 1
            if sample_count % n_check == 0 and self.parameters["print_level"] > 0:
                print("{0:2.1f} % completed, Acceptance ratio {1:2.1f} %".format(
                    float(sample_count) / float(number_of_samples) * 100,
                    float(naccept) / float(sample_count) * 100))
                print("Process has taken{0:6.0f} s".format(time.time() - t0))
        tracer.save()
        return naccept

    def consume_random(self):
        number_of_samples = self.parameters["number_of_samples"]
        for ii in range(number_of_samples):
            self.kernel.consume_random()

class MultiLevelDelayedAcceptanceMCMC:
    def __init__(self, kernels : list):
        """
        :param kernels: list of kernels
        """
        self.kernels = kernels
        self.n_levels = len(kernels)
        assert isinstance(kernels[-1], DelayedAcceptanceKernel)
        self.parameters = {}
        self.parameters["number_of_samples"] = None
        self.parameters["print_progress"] = 20
        self.parameters["print_level"] = 1

    def run(self, m0, qoi=None, tracer=None):

        if qoi is None:
            qoi = hp.NullQoi()
        if tracer is None:
            tracer = NullTracer()
        assert self.parameters["number_of_samples"] is not None, "The number of samples is not set"
        number_of_samples = self.parameters["number_of_samples"]

        current = [self.kernels[i].generate_sample() for i in range(self.n_levels)]
        proposed = [self.kernels[i].generate_sample() for i in range(self.n_levels)]

        for level in range(self.n_levels):
            current[level].assign_paramaeter(m0)
            self.kernels[level].init_sample(current[level])

        if self.parameters["print_level"] > 0:
            print("Generate {0} samples".format(number_of_samples))
        sample_count = 0
        naccept = 0
        n_check = number_of_samples // self.parameters["print_progress"]
        t0 = time.time()
        while (sample_count < number_of_samples):
            accept = -np.ones(self.n_levels)
            for level in range(self.n_levels):
                self.kernels[level].sample(current[level], proposed[level])
                self.kernels[level].init_sample(proposed[level])
                if level == 0:
                    accept[level] = self.kernels[level].accept_or_reject(current[level], proposed[level])
                else:
                    accept[level] = self.kernels[level].accept_or_reject(current[level-1:level+1], proposed[level-1:level+1])
                if accept[level] == 1:
                    if level < self.n_levels - 1:
                        proposed[level+1].assign(proposed[level])
                    else:
                        for level in range(self.n_levels):
                            current[level].assign(proposed[level])
                else:
                    break
            naccept += np.min(np.abs(accept))
            q = qoi.eval([current[-1].u, current[-1].m])
            tracer.append(current[-1], q, accept)
            sample_count += 1
            if sample_count % n_check == 0 and self.parameters["print_level"] > 0:
                print("{0:2.1f} % completed, Acceptance ratio {1:2.1f} %".format(
                    float(sample_count) / float(number_of_samples) * 100,
                    float(naccept) / float(sample_count) * 100))
                print("Process has taken{0:6.0f} s".format(time.time() - t0))
        tracer.save()
        return naccept

    def consume_random(self):
        number_of_samples = self.parameters["number_of_samples"]
        for ii in range(number_of_samples):
            for level in range(self.n_levels):
                self.kernels[level].consume_random()
                if level<self.n_levels-1 and (self.kernels[level].reduced() and not self.kernels[level+1].reduced()):
                    self.kernels[level+1].consume_random()
