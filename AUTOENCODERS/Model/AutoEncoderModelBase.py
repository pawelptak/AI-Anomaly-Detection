from tensorflow.keras import Sequential


class AutoEncoderModelBase:
    def build_model(self) -> Sequential:
        pass
