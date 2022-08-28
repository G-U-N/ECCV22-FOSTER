from models.foster import FOSTER
from models.rmm import RMM_FOSTER
def get_model(model_name, args):
    name = model_name.lower()
    if name == "foster":
        return FOSTER(args)
    elif name == "rmm-foster":
        return RMM_FOSTER(args)
    else:
        assert 0, "Not Implemented!"
