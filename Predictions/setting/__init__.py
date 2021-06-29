from Predictions.setting.setting import Setting

STANDARD = Setting(10, "train-test", True, False, 70, 10)

TAX = Setting(None, "train-test", False, True, 66)
LIN = Setting(5, "train-test", False, False, 70, filter_cases=5)
CAMARGO = Setting(5, "test-train", False, True, 70)
DIMAURO = Setting(None, "k-fold", False, True, 80, train_k=3)
PASQUADIBISCEGLIE = Setting(10, "train-test", False, True, 66)
TAYMOURI = Setting(10, "train-test", True, True, 80)
SDL = Setting(10, "train-test", True, False, 70)
DBN = Setting(5, "test-train", False, True, 70, filter_cases=5)

ALL = [TAX, CAMARGO, LIN, DIMAURO, PASQUADIBISCEGLIE, TAYMOURI, DBN, SDL]