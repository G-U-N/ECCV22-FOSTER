from models.foster import FOSTER
def get_model(model_name, args):
    name = model_name.lower()
    if name == "foster":
        return FOSTER(args)
    else:
        assert 0, "Not Implemented!"
