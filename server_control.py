from ServerControlStub import ServerControlStub
#from i24_sys import ServerControlStub
import time
from i24_logger.log_writer import logger




# CODEWRITER TODO - import your process targets here such that these functions can be directly passed as the target to mp.Process or mp.Pool
from execute import main
from src.scene.homography          import HomographyWrapper,Homography


def dummy_function(arg1,arg2,kwarg1 = None):
    logger.set_name("DUMMY {}".format(kwarg1))
    while True:
        logger.debug("This is what happens when you start a process!")
        time.sleep(5)

# CODEWRITER TODO - add your process targets to register_functions
register_functions = [dummy_function,main]
name_to_process = dict([(fn.__name__, fn) for fn in register_functions])



class ServerControl(ServerControlStub):
    
    def get_additional_args(self):
        # CODEWRITER TODO - Implement any shared variables (queues etc. here)
        # each entry is a tuple (args,kwargs) (list,dict)
        # to be added to process args/kwargs and key is process_name
        
        return {}



if __name__ == "__main__":
    s = ServerControl(name_to_process)
