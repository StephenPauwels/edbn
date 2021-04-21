from setting.setting import Setting

STANDARD = Setting(10, "train-test", True, False, 70, 10)

TAX = Setting(None, "train-test", False, True, 66)
LIN = Setting(5, "train-test", False, False, 70)
CAMARGO = Setting(5, "test-train", False, True, 70)
DIMAURO = Setting(None, "random", False, True, 80)
PASQUADIBISCEGLIE = Setting(None, "train-test", False, True, 66)
TAYMOURI = Setting(5, "train-test", True, True, 80)

ALL = [TAX, LIN, CAMARGO, DIMAURO, PASQUADIBISCEGLIE, TAYMOURI, STANDARD]