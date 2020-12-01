"""Utils for training, including time and file operation"""

import os
import time

get_current_time = lambda: time.strftime("%Y%m%d-%H%M%S", time.localtime())
filesetting = lambda path: os.makedirs(path) if not os.path.exists(path) else None
