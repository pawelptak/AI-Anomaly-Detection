from tensorflow.keras import Sequential

"""
The base class for all models implementations.
"""


class AutoEncoderModelBase:
    def build_model(self) -> Sequential:
        pass
