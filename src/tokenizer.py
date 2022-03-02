class Tokenizer:

    def __init__(self, segmenter):
        self.sentences = segmenter.apply_segmentation(annotated=False, debug=False)

