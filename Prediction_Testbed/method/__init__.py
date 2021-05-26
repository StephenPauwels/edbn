from method.method import Method


def get_method(method_name):
    if method_name == "SDL":
        from method import sdl
        return Method("SDL", sdl.train, sdl.test, sdl.update, {"epochs": 200, "early_stop": 10})
    elif method_name == "DBN":
        from Predictions import edbn_adapter as edbn
        return Method("DBN", edbn.train, edbn.test, edbn.update)
    elif method_name == "CAMARGO":
        from RelatedMethods.Camargo import adapter as camargo
        return Method("Camargo", camargo.train, camargo.test, camargo.update, {"epochs": 200, "early_stop": 10})
    elif method_name == "DIMAURO":
        from RelatedMethods.DiMauro import adapter as dimauro
        return Method("Di Mauro", dimauro.train, dimauro.test, dimauro.update, {"epochs": 200, "early_stop": 10})
    elif method_name == "LIN":
        from RelatedMethods.Lin import adapter as lin
        return Method("Lin", lin.train, lin.test, lin.update, {"epochs": 200, "early_stop": 10})
    elif method_name == "PASQUADIBISCEGLIE":
        from RelatedMethods.Pasquadibisceglie import adapter as pasquadibisceglie
        return Method("Pasquadibisceglie", pasquadibisceglie.train, pasquadibisceglie.test,
                      pasquadibisceglie.update, {"epochs": 200, "early_stop": 10})
    elif method_name == "TAX":
        from RelatedMethods.Tax import adapter as tax
        return Method("Tax", tax.train, tax.test, tax.update, {"epochs": 200, "early_stop": 10})
    elif method_name == "TAYMOURI":
        from RelatedMethods.Taymouri import adapter as taymouri
        return Method("Taymouri", taymouri.train, taymouri.test, {"epoch": 10})


ALL = ["SDL", "DBN", "CAMARGO", "DIMAURO", "LIN", "PASQUADIBISCEGLIE", "TAX", "TAYMOURI"]
