from typing import Any, Dict, List
import logging
from ..types.config import PreprocessingConfig

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data preprocessing operations."""

    def process(self, data: Any, config: PreprocessingConfig) -> Any:
        """Apply preprocessing operations to data.

        Args:
            data: Input data to process
            config: Preprocessing configuration

        Returns:
            Processed data
        """
        try:
            # Apply preprocessing based on data type and config
            if self._is_tabular_data(data):
                return self._process_tabular(data, config)
            elif self._is_image_data(data):
                return self._process_image(data, config)
            elif self._is_text_data(data):
                return self._process_text(data, config)
            else:
                logger.warning("Unknown data type, applying generic preprocessing")
                return self._process_generic(data, config)
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

    def _is_tabular_data(self, data: Any) -> bool:
        """Check if data is tabular (DataFrame, CSV-like)."""
        try:
            import pandas as pd

            return isinstance(data, pd.DataFrame)
        except ImportError:
            return False

    def _is_image_data(self, data: Any) -> bool:
        """Check if data is image data."""
        try:
            from PIL import Image

            return isinstance(data, Image.Image) or isinstance(data, list)
        except ImportError:
            return False

    def _is_text_data(self, data: Any) -> bool:
        """Check if data is text data."""
        return isinstance(data, (str, list)) and not self._is_image_data(data)

    def _process_tabular(self, data: Any, config: PreprocessingConfig) -> Any:
        """Process tabular data (pandas DataFrame)."""
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            return data

        processed_data = data.copy()

        # Filter columns if specified
        if config.filter_columns:
            available_columns = [
                col for col in config.filter_columns if col in processed_data.columns
            ]
            if available_columns:
                processed_data = processed_data[available_columns]
                logger.debug(f"Filtered to columns: {available_columns}")

        # Normalize data
        if config.normalize:
            processed_data = self._normalize_dataframe(processed_data)
            logger.debug("Applied normalization")

        # Shuffle data
        if config.shuffle:
            processed_data = processed_data.sample(frac=1).reset_index(drop=True)
            logger.debug("Shuffled data")

        # Split data if ratio provided
        if config.split_ratio and len(config.split_ratio) > 1:
            return self._split_dataframe(processed_data, config.split_ratio)

        return processed_data

    def _process_image(self, data: Any, config: PreprocessingConfig) -> Any:
        """Process image data."""
        try:
            from PIL import Image
        except ImportError:
            logger.warning("PIL not available for image processing")
            return data

        def process_single_image(img):
            if not isinstance(img, Image.Image):
                return img

            processed_img = img

            # Resize if specified
            if config.resize:
                processed_img = processed_img.resize(tuple(config.resize))
                logger.debug(f"Resized image to {config.resize}")

            # Convert to grayscale
            if config.grayscale:
                processed_img = processed_img.convert("L")
                logger.debug("Converted to grayscale")

            # Apply augmentation
            if config.augmentation:
                processed_img = self._apply_image_augmentation(
                    processed_img, config.augmentation
                )

            return processed_img

        if isinstance(data, list):
            return [process_single_image(img) for img in data]
        else:
            return process_single_image(data)

    def _process_text(self, data: Any, config: PreprocessingConfig) -> Any:
        """Process text data."""
        if isinstance(data, str):
            return self._process_single_text(data, config)
        elif isinstance(data, list):
            return [
                self._process_single_text(text, config)
                for text in data
                if isinstance(text, str)
            ]
        else:
            return data

    def _process_single_text(self, text: str, config: PreprocessingConfig) -> Any:
        """Process a single text string."""
        processed_text = text

        # Tokenization
        if config.tokenization:
            processed_text = self._tokenize_text(processed_text)

        # Remove stopwords
        if config.remove_stopwords:
            processed_text = self._remove_stopwords(processed_text)

        # Stemming
        if config.stemming:
            processed_text = self._apply_stemming(processed_text)

        return processed_text

    def _process_generic(self, data: Any, config: PreprocessingConfig) -> Any:
        """Apply generic preprocessing operations."""
        processed_data = data

        # Shuffle if it's a list
        if config.shuffle and isinstance(data, list):
            import random

            processed_data = data.copy()
            random.shuffle(processed_data)
            logger.debug("Shuffled list data")

        return processed_data

    def _normalize_dataframe(self, df) -> Any:
        """Normalize numeric columns in DataFrame."""
        numeric_columns = df.select_dtypes(include=["number"]).columns
        df_normalized = df.copy()

        for col in numeric_columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:  # Avoid division by zero
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)

        return df_normalized

    def _split_dataframe(self, df, split_ratio: List[float]) -> Dict[str, Any]:
        """Split DataFrame according to ratios."""
        if sum(split_ratio) != 1.0:
            # Normalize ratios
            total = sum(split_ratio)
            split_ratio = [r / total for r in split_ratio]

        n = len(df)
        splits = {}
        start_idx = 0

        split_names = ["train", "validation", "test"][: len(split_ratio)]

        for i, (name, ratio) in enumerate(zip(split_names, split_ratio)):
            if i == len(split_ratio) - 1:  # Last split gets remaining data
                end_idx = n
            else:
                end_idx = start_idx + int(n * ratio)

            splits[name] = df.iloc[start_idx:end_idx].reset_index(drop=True)
            start_idx = end_idx

        logger.debug(
            f"Split data into: {[f'{k}({len(v)})' for k, v in splits.items()]}"
        )
        return splits

    def _apply_image_augmentation(self, img, augmentation_config: Dict[str, bool]):
        """Apply image augmentation operations."""
        try:
            from PIL import Image
            import random
        except ImportError:
            logger.warning("PIL not available for image augmentation")
            return img

        augmented_img = img

        # Flip operations
        if augmentation_config.get("flip", False):
            if random.random() > 0.5:
                augmented_img = augmented_img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                augmented_img = augmented_img.transpose(Image.FLIP_TOP_BOTTOM)

        # Rotation
        if augmentation_config.get("rotate", False):
            angle = random.randint(-30, 30)
            augmented_img = augmented_img.rotate(angle)

        return augmented_img

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization - can be enhanced with NLTK or spaCy
        import re

        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def _remove_stopwords(self, text) -> Any:
        """Remove common stopwords from text."""
        # Basic English stopwords - can be enhanced with NLTK
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "the",
            "this",
            "but",
            "they",
            "have",
            "had",
            "what",
            "said",
            "each",
            "which",
            "she",
            "do",
            "how",
            "their",
            "if",
            "up",
            "out",
            "many",
            "then",
            "them",
            "these",
            "so",
            "some",
            "her",
            "would",
            "make",
            "like",
            "into",
            "him",
            "time",
            "two",
            "more",
            "go",
            "no",
            "way",
            "could",
            "my",
            "than",
            "first",
            "been",
            "call",
            "who",
            "oil",
            "sit",
            "now",
            "find",
            "down",
            "day",
            "did",
            "get",
            "come",
            "made",
            "may",
            "part",
        }

        if isinstance(text, list):
            return [word for word in text if word.lower() not in stopwords]
        elif isinstance(text, str):
            words = text.split()
            filtered_words = [word for word in words if word.lower() not in stopwords]
            return " ".join(filtered_words)
        else:
            return text

    def _apply_stemming(self, text) -> Any:
        """Apply basic stemming to text."""

        # Very basic stemming - can be enhanced with NLTK Porter Stemmer
        def simple_stem(word: str) -> str:
            # Remove common suffixes
            suffixes = ["ing", "ed", "er", "est", "ly", "s"]
            for suffix in suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    return word[: -len(suffix)]
            return word

        if isinstance(text, list):
            return [simple_stem(word) for word in text]
        elif isinstance(text, str):
            words = text.split()
            stemmed_words = [simple_stem(word) for word in words]
            return " ".join(stemmed_words)
        else:
            return text
