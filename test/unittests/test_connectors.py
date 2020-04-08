import unittest
from mozaik.connectors import vision
import numpy
import numpy.linalg
import logging

class TestV1CorrelationBasedConnectivity(unittest.TestCase):
    def setUp(self):
        self.number_of_tests = 100
    
    @staticmethod
    def generate_a_list_of_gabor_parameters(length):
        params = []
        for i in range(0,length):
            K = 0.1+numpy.random.rand()*9.9
            a = 1/(1+numpy.random.rand()*9)
            b = 1/(1+numpy.random.rand()*9)
            F = 0.1+numpy.random.rand()*0.3
            x0 =-5+numpy.random.rand()*5
            y0 =-5+numpy.random.rand()*5
            omega = numpy.random.rand()*numpy.pi
            theta = omega # in vision we always work with the special case where the orientation of the gaussian is the same as orientation along which the grating varies
            P = numpy.random.rand()*numpy.pi*2
            params.append([K, a, b, x0, y0, theta, F, omega, P])
        return params    
   
    @staticmethod
    def generate_a_list_of_gabor_relative_parameters(length):
        params = []
        for i in range(0,length):
            x0 = -5+numpy.random.rand()*5
            y0 = -5+numpy.random.rand()*5
            F = 0.1+numpy.random.rand()*0.3
            orr = numpy.random.rand()*numpy.pi
            P = numpy.random.rand()*numpy.pi*2
            size = numpy.random.rand()*10
            aspect_ratio = 0.2 + numpy.random.rand()*5
            params.append([size,x0,y0,aspect_ratio,orr,F,P])
        return params    
   
    
    @staticmethod
    def gabor_connectivity_gabor(width,posx,posy,ar,orr,freq,phase):
        XX, YY = numpy.ogrid[-40:40:2000j,-40:40:2000j]
        return vision.gabor(XX, YY,posx, posy, orr, freq, phase, width, ar)

    @staticmethod
    def real_gabor_with_relative_parameters(width,posx,posy,ar,orr,freq,phase):
        return TestV1CorrelationBasedConnectivity.real_gabor(1.0, 1/(numpy.sqrt(2*numpy.pi)*width),ar/(numpy.sqrt(2*numpy.pi)*width), posx, posy, orr, freq, orr, phase-numpy.pi*2*freq*(posx*numpy.cos(orr)+posy*numpy.sin(orr)))

    
    
    @staticmethod
    def real_gabor(K, a, b, x0, y0, theta, F, omega, P):
        #a, b    - gaussian widths (1/over)
        #x0, y0  - centre of gaussian
        #F, P    - spatial frequency and phase of grating
        #theta,omega  - orientation angles of gaussian and grating
        #K       - scaler     
        XX, YY = numpy.ogrid[-40:40:2000j,-40:40:2000j]
        x_r = (XX-x0)*numpy.cos(theta)+(YY-y0)*numpy.sin(theta)
        y_r = -(XX-x0)*numpy.sin(theta)+(YY-y0)*numpy.cos(theta)
        gaussian = K*numpy.exp(-numpy.pi*(a**2*x_r**2+b**2*y_r**2))
        complex_grating = numpy.exp(1j*2*numpy.pi*F*(XX*numpy.cos(omega)+YY*numpy.sin(omega))+1j*P)
        return numpy.real(gaussian*complex_grating)


    def test_integral_of_gabor_multiplication(self):
        for p1,p2 in zip(self.generate_a_list_of_gabor_parameters(self.number_of_tests),self.generate_a_list_of_gabor_parameters(self.number_of_tests)):
            iogm1 = numpy.dot(self.real_gabor(*p1).flatten(), self.real_gabor(*p2).flatten())*(80./2000)**2
            iogm2 = numpy.array(vision.V1CorrelationBasedConnectivity.integral_of_gabor_multiplication(*(p1+p2)))[0][0]
            self.assertAlmostEqual(iogm1,iogm2,2,"The integral of multiplication of two gabors with parameters %f,%f,%f,%f,%f,%f,%f,%f,%f and %f,%f,%f,%f,%f,%f,%f,%f,%f does not match. Empirical value: %g, analytical value: %g." % (p1[0],p1[1],p1[2],p1[3],p1[4],p1[5],p1[6],p1[7],p1[8],p2[0],p2[1],p2[2],p2[3],p2[4],p2[5],p2[6],p2[7],p2[8],iogm1,iogm2))

    def test_gabor_correlation(self):
        for p1,p2 in zip(self.generate_a_list_of_gabor_parameters(self.number_of_tests),self.generate_a_list_of_gabor_parameters(self.number_of_tests)):
            g1 = self.real_gabor(*p1).flatten()
            g2 = self.real_gabor(*p2).flatten()
            iogm1 = numpy.dot(g1-numpy.mean(g1),g2-numpy.mean(g2))/len(g1)/(numpy.std(g1)*numpy.std(g2))
            iogm2 = vision.V1CorrelationBasedConnectivity.gabor_correlation(*(p1+p2))
            self.assertAlmostEqual(iogm1,iogm2,2,"The integral of multiplication of two gabors with parameters %f,%f,%f,%f,%f,%f,%f,%f,%f and %f,%f,%f,%f,%f,%f,%f,%f,%f does not match. Empirical value: %g, analytical value: %g." % (p1[0],p1[1],p1[2],p1[3],p1[4],p1[5],p1[6],p1[7],p1[8],p2[0],p2[1],p2[2],p2[3],p2[4],p2[5],p2[6],p2[7],p2[8],iogm1,iogm2))

    def test_gabor_correlation_with_gaussian_used_for_connections(self):
        for p1,p2 in zip(self.generate_a_list_of_gabor_relative_parameters(self.number_of_tests),self.generate_a_list_of_gabor_relative_parameters(self.number_of_tests)):
            g1 = self.gabor_connectivity_gabor(*p1).flatten()
            g2 = self.gabor_connectivity_gabor(*p2).flatten()
            iogm1 = numpy.dot(g1-numpy.mean(g1),g2-numpy.mean(g2))/len(g1)/(numpy.std(g1)*numpy.std(g2))
            iogm2 = vision.V1CorrelationBasedConnectivity.gabor_correlation_rescaled_parammeters(*(p1+p2))
            self.assertAlmostEqual(iogm1,iogm2,2,"The integral of multiplication of two gabors with parameters %f,%f,%f,%f,%f,%f,%f and %f,%f,%f,%f,%f,%f,%f does not match. Empirical value: %g, analytical value: %g." % (p1[0],p1[1],p1[2],p1[3],p1[4],p1[5],p1[6],p2[0],p2[1],p2[2],p2[3],p2[4],p2[5],p2[6],iogm1,iogm2))

    def test_integral_of_gabor_multiplication_vectorized(self):            
        param1,param2 = self.generate_a_list_of_gabor_parameters(self.number_of_tests),self.generate_a_list_of_gabor_parameters(self.number_of_tests)
        iogm2 = []
        iogm1 = vision.V1CorrelationBasedConnectivity.integral_of_gabor_multiplication_vectorized(*(tuple(numpy.array(param1).T)+tuple(numpy.array(param2).T)))
        for p1,p2 in zip(param1,param2):
            iogm2.append(numpy.array(vision.V1CorrelationBasedConnectivity.integral_of_gabor_multiplication(*(p1+p2)))[0][0])
        
        for i,(a,b) in enumerate(zip(iogm1,iogm2)):
            self.assertAlmostEqual(a,b,8,"The integral of multiplication of two gabors with parameters %f,%f,%f,%f,%f,%f,%f,%f,%f and %f,%f,%f,%f,%f,%f,%f,%f,%f does not match. Matrix version: %g, vectorized version: %g." % (param1[i][0],param1[i][1],param1[i][2],param1[i][3],param1[i][4],param1[i][5],param1[i][6],param1[i][7],param1[i][8],param2[i][0],param2[i][1],param2[i][2],param2[i][3],param2[i][4],param2[i][5],param2[i][6],param2[i][7],param2[i][8],a,b))

    
        
class TestConnector(unittest.TestCase):
    pass

class TestMozaikConnector(unittest.TestCase):
    pass

class TestSpecificArborization(unittest.TestCase):
    pass


class TestSpecificProbabilisticArborization(unittest.TestCase):
    pass


class TestDistanceDependentProbabilisticArborization(unittest.TestCase):
    pass


class TestExponentialProbabilisticArborization(unittest.TestCase):
    pass


class TestUniformProbabilisticArborization(unittest.TestCase):
    pass


class TestGaborConnector(unittest.TestCase):
    pass


class TestModularConnectorFunction(unittest.TestCase):
    pass


class TestConstantModularConnectorFunction(unittest.TestCase):
    pass


class TestPyNNDistributionConnectorFunction(unittest.TestCase):
    pass


class TestDistanceDependentModularConnectorFunction(unittest.TestCase):
    pass


class TestGaussianDecayModularConnectorFunction(unittest.TestCase):
    pass


class TestExponentialDecayModularConnectorFunction(unittest.TestCase):
    pass


class TestLinearModularConnectorFunction(unittest.TestCase):
    pass


class TestHyperbolicModularConnectorFunction(unittest.TestCase):
    pass


class TestMapDependentModularConnectorFunction(unittest.TestCase):
    pass


class TestV1PushPullArborization(unittest.TestCase):
    pass


class TestModularConnector(unittest.TestCase):
    pass


class TestModularSamplingProbabilisticConnector(unittest.TestCase):
    pass


class TestModularSingleWeightProbabilisticConnector(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()
