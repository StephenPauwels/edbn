from setting.setting import Setting

STANDARD = Setting(10, 70, "train-test", True, False)

TAX = Setting(None, 66, "train-test", False, True)
LIN = Setting(5, 70, "train-test", False, False)
CAMARGO = Setting(5, 70, "test-train", False, True)
DIMAURO = Setting(None, 80, "random", False, True)
PASQUADIBISCEGLIE = Setting(None, 66, "train-test", False, True)
TAYMOURI = Setting(5, 80, "train-test", True, True)

ALL = [TAX, LIN, CAMARGO, DIMAURO, PASQUADIBISCEGLIE, TAYMOURI, STANDARD]