import os

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = '<|video_pad|>'
DEFAULT_IMAGE_START_TOKEN = "<|vision_start|>"
DEFAULT_IMAGE_END_TOKEN = "<|vision_end|>"

# Video Constants
MAX_PIXELS = int(os.environ['MAX_PIXELS']) if 'MAX_PIXELS' in os.environ else 448*448
MIN_PIXELS = int(os.environ['MIN_PIXELS']) if 'MIN_PIXELS' in os.environ else 112*112
MIN_FRAMES = int(os.environ['MIN_FRAMES']) if 'MIN_FRAMES' in os.environ else 16
MAX_FRAMES = int(os.environ['MAX_FRAMES']) if 'MAX_FRAMES' in os.environ else 16

# DETECT Prompt
DEC_PROMPT = """Detect objects and their 3d boxes. The reference code is as followed:
@dataclass
class Bbox:
   class: str
   position_x: int
   position_y: int
   position_z: int
   angle_z: float
   scale_x: int
   scale_y: int
   scale_z: int
"""

SCANREF_PROMPT = """Localize the object and its 3d boxes according to the description given you.
You should Localize the <object_name>. There is only one object you should localize.
Description: <object_description>
The reference code is as followed:
@dataclass
class Bbox:
   class: str
   position_x: int
   position_y: int
   position_z: int
   angle_z: float
   scale_x: int
   scale_y: int
   scale_z: int
"""

MULTI3DREF_PROMPT = """Localize the object and its 3d boxes according to the description given you.
You should Localize the <object_name>. There may be one or more objects you should localize.
Description: <object_description>
The reference code is as followed:
@dataclass
class Bbox:
   class: str
   position_x: int
   position_y: int
   position_z: int
   angle_z: float
   scale_x: int
   scale_y: int
   scale_z: int
"""
