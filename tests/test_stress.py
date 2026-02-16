import unittest
from src.logic import StressCalculator

class MockBlendshape:
    def __init__(self, name, score):
        self.category_name = name
        self.score = score

class TestStressCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = StressCalculator()
        
    def test_stress_calculation_low(self):
        # Neutral face
        blendshapes = [
            MockBlendshape('browDownLeft', 0.0),
            MockBlendshape('browDownRight', 0.0),
            MockBlendshape('eyeSquintLeft', 0.0),
            MockBlendshape('eyeSquintRight', 0.0),
            MockBlendshape('mouthPressLeft', 0.0),
            MockBlendshape('mouthPressRight', 0.0)
        ]
        score = self.calc.calculate_score(blendshapes)
        self.assertAlmostEqual(score, 0.0)
        
    def test_stress_calculation_high(self):
        # Max stress indicators
        blendshapes = [
            MockBlendshape('browDownLeft', 1.0),
            MockBlendshape('browDownRight', 1.0),
            MockBlendshape('eyeSquintLeft', 1.0),
            MockBlendshape('eyeSquintRight', 1.0),
            MockBlendshape('mouthPressLeft', 1.0),
            MockBlendshape('mouthPressRight', 1.0)
        ]
        score = self.calc.calculate_score(blendshapes)
        # 0.4 + 0.3 + 0.3 = 1.0
        self.assertAlmostEqual(score, 1.0)

    def test_stress_calculation_mixed(self):
        # Half brow, full squint, no lip
        blendshapes = [
            MockBlendshape('browDownLeft', 0.5),
            MockBlendshape('browDownRight', 0.5), # 0.5 avg * 0.4 = 0.2
            MockBlendshape('eyeSquintLeft', 1.0),
            MockBlendshape('eyeSquintRight', 1.0), # 1.0 avg * 0.3 = 0.3
            MockBlendshape('mouthPressLeft', 0.0),
            MockBlendshape('mouthPressRight', 0.0) # 0.0 avg * 0.3 = 0.0
        ]
        score = self.calc.calculate_score(blendshapes)
        self.assertAlmostEqual(score, 0.5)

if __name__ == '__main__':
    unittest.main()
