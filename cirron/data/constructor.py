from typing import Dict, List, Any, Optional, Union
import logging
from ..types.config import DataConfig, DataSourceConfig, dict_to_data_config
from .processors import DataProcessor
from .sources import DataSourceFactory

logger = logging.getLogger(__name__)


class CirronData:
    """Enhanced data constructor for Cirron SDK.

    Provides a config-based approach to data source management,
    preprocessing, and integration similar to how CirronModel works for models.

    Examples:
        >>> data_config = {
        ...     "data_sources": [
        ...         {
        ...             "source_name": "training_data",
        ...             "source_type": "local",
        ...             "path": "/data/train.csv",
        ...             "format": "csv",
        ...             "preprocessing": {"normalize": True, "shuffle": True}
        ...         }
        ...     ]
        ... }
        >>> data = CirronData(data_config)
        >>> processed_data = data.load_and_process()
    """

    def __init__(
        self,
        data_config: Union[Dict[str, Any], DataConfig],
        cirron_instance: Optional["Cirron"] = None,
        **kwargs,
    ):
        """Initialize CirronData with configuration.

        Args:
            data_config: Data configuration dictionary or DataConfig object
            cirron_instance: Parent Cirron instance for context
            **kwargs: Additional configuration options
        """
        self.cirron_instance = cirron_instance
        self._raw_config = data_config

        # Convert dict to DataConfig if needed
        if isinstance(data_config, dict):
            self.config = dict_to_data_config(data_config)
        else:
            self.config = data_config

        # Initialize components
        self.data_sources = {}
        self.processor = DataProcessor()
        self.source_factory = DataSourceFactory()

        # Build data sources
        self._build_data_sources()

        logger.info(
            f"CirronData initialized with {len(self.config.data_sources)} data sources"
        )

    def _build_data_sources(self):
        """Build data source objects from configuration."""
        for source_config in self.config.data_sources:
            try:
                source = self.source_factory.create_source(source_config)
                self.data_sources[source_config.source_name] = source
                logger.debug(f"Created data source: {source_config.source_name}")
            except Exception as e:
                logger.error(
                    f"Failed to create data source {source_config.source_name}: {e}"
                )
                raise

    def load_data(self, source_name: Optional[str] = None) -> Any:
        """Load data from specified source or all sources.

        Args:
            source_name: Specific source to load, or None for all sources

        Returns:
            Loaded data
        """
        if source_name:
            if source_name not in self.data_sources:
                raise ValueError(f"Data source '{source_name}' not found")
            return self.data_sources[source_name].load()
        else:
            # Load all sources
            data = {}
            for name, source in self.data_sources.items():
                data[name] = source.load()
            return data

    def process_data(self, data: Any, source_name: Optional[str] = None) -> Any:
        """Apply preprocessing to data.

        Args:
            data: Raw data to process
            source_name: Source name to get preprocessing config

        Returns:
            Processed data
        """
        if source_name:
            # Find the source config for preprocessing
            source_config = next(
                (s for s in self.config.data_sources if s.source_name == source_name),
                None,
            )
            if source_config and source_config.preprocessing:
                return self.processor.process(data, source_config.preprocessing)

        return data

    def load_and_process(self, source_name: Optional[str] = None) -> Any:
        """Load data and apply preprocessing in one step.

        Args:
            source_name: Specific source to process, or None for all sources

        Returns:
            Loaded and processed data
        """
        if source_name:
            raw_data = self.load_data(source_name)
            return self.process_data(raw_data, source_name)
        else:
            # Process all sources
            processed_data = {}
            for name in self.data_sources.keys():
                raw_data = self.load_data(name)
                processed_data[name] = self.process_data(raw_data, name)
            return processed_data

    def get_source_info(self, source_name: str) -> Dict[str, Any]:
        """Get information about a data source.

        Args:
            source_name: Name of the data source

        Returns:
            Dictionary with source information
        """
        if source_name not in self.data_sources:
            raise ValueError(f"Data source '{source_name}' not found")

        source_config = next(
            s for s in self.config.data_sources if s.source_name == source_name
        )

        return {
            "name": source_config.source_name,
            "type": source_config.source_type,
            "format": source_config.format,
            "description": source_config.description,
            "preprocessing": (
                source_config.preprocessing.__dict__
                if source_config.preprocessing
                else None
            ),
            "path": getattr(source_config, "path", None),
            "cloud_provider": getattr(source_config, "cloud_provider", None),
            "bucket_name": getattr(source_config, "bucket_name", None),
        }

    def list_sources(self) -> List[str]:
        """List all available data source names.

        Returns:
            List of data source names
        """
        return list(self.data_sources.keys())

    def save_processed_data(self, data: Any, source_name: Optional[str] = None) -> bool:
        """Save processed data to target destination.

        Args:
            data: Processed data to save
            source_name: Source name for context

        Returns:
            True if successful, False otherwise
        """
        if not self.config.target_destination:
            logger.warning("No target destination configured")
            return False

        try:
            # TODO: Implement actual saving logic based on target destination config
            logger.info(
                f"Saving data to {self.config.target_destination.type} destination"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            return False

    def refresh_data(self) -> bool:
        """Refresh data according to refresh configuration.

        Returns:
            True if successful, False otherwise
        """
        if not self.config.data_refresh:
            logger.warning("No data refresh configuration found")
            return False

        try:
            # TODO: Implement actual refresh logic
            logger.info(
                f"Refreshing data with {self.config.data_refresh.interval} interval"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to refresh data: {e}")
            if self.config.data_refresh.notify_on_failure:
                logger.error("Notification would be sent on failure")
            return False

    def __repr__(self) -> str:
        """String representation of CirronData."""
        return f"CirronData(sources={len(self.data_sources)}, config={self.config})"
