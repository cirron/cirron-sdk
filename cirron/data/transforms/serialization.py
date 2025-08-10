"""
Drift-aware serialization system for transform artifacts.

This module provides versioned serialization and loading of fitted transforms
with schema validation, drift detection, and comprehensive metadata tracking.
"""

import json
import pickle
import hashlib
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)


class TransformArtifact:
    """Versioned artifact container for fitted transforms.
    
    Provides comprehensive serialization with schema tracking, drift detection,
    and versioning capabilities for production deployment.
    """
    
    def __init__(
        self,
        transform: Any,
        artifact_id: Optional[str] = None,
        version: str = "1.0.0",
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize transform artifact.
        
        Args:
            transform: Fitted transform object to serialize
            artifact_id: Unique identifier for the artifact
            version: Version string (semantic versioning recommended)
            description: Human-readable description
            metadata: Additional metadata dictionary
        """
        if not hasattr(transform, '_is_fitted') or not transform._is_fitted:
            raise ValueError("Transform must be fitted before creating artifact")
        
        self.transform = transform
        self.artifact_id = artifact_id or self._generate_artifact_id()
        self.version = version
        self.description = description or f"{transform.__class__.__name__} v{version}"
        self.metadata = metadata or {}
        
        # Extract schema and fingerprint
        self.schema = self._extract_schema()
        self.fingerprint = self._generate_fingerprint()
        self.creation_time = datetime.datetime.utcnow().isoformat()
    
    def _generate_artifact_id(self) -> str:
        """Generate unique artifact ID."""
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        class_name = self.transform.__class__.__name__
        random_suffix = hashlib.md5(str(datetime.datetime.utcnow()).encode()).hexdigest()[:8]
        return f"{class_name}_{timestamp}_{random_suffix}"
    
    def _extract_schema(self) -> Dict[str, Any]:
        """Extract schema information from fitted transform."""
        schema = {
            'transform_class': self.transform.__class__.__name__,
            'transform_module': self.transform.__class__.__module__,
            'parameters': self.transform.get_params(),
        }
        
        # Add fitted schema if available
        if hasattr(self.transform, '_fitted_schema') and self.transform._fitted_schema:
            schema['fitted_schema'] = self.transform._fitted_schema
        
        # Add vocabulary information for encoders
        if hasattr(self.transform, 'get_vocabulary'):
            try:
                schema['vocabularies'] = self.transform.get_vocabulary()
            except:
                pass  # Not all encoders may implement this
        
        # Add fitted parameters
        if hasattr(self.transform, 'get_fitted_params'):
            try:
                schema['fitted_parameters'] = self.transform.get_fitted_params()
            except:
                pass
        
        return schema
    
    def _generate_fingerprint(self) -> str:
        """Generate data fingerprint for drift detection."""
        fingerprint_data = {
            'transform_class': self.transform.__class__.__name__,
            'parameters': self.transform.get_params(),
        }
        
        # Add schema fingerprint if available
        if 'fitted_schema' in self.schema:
            fitted_schema = self.schema['fitted_schema']
            fingerprint_data.update({
                'columns': fitted_schema.get('columns', []),
                'dtypes': fitted_schema.get('dtypes', {}),
                'shape': fitted_schema.get('shape', ()),
            })
        
        # Add vocabulary fingerprint for encoders
        if 'vocabularies' in self.schema:
            vocab_hash = {}
            for col, vocab in self.schema['vocabularies'].items():
                vocab_str = ','.join(sorted(str(v) for v in vocab))
                vocab_hash[col] = hashlib.md5(vocab_str.encode()).hexdigest()[:16]
            fingerprint_data['vocabulary_hashes'] = vocab_hash
        
        # Create combined hash
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()
    
    def save(self, directory: Union[str, Path], compress: bool = True) -> Path:
        """Save artifact to directory.
        
        Args:
            directory: Directory to save artifact in
            compress: Whether to compress the transform state
            
        Returns:
            Path to the saved artifact directory
        """
        artifact_dir = Path(directory) / f"{self.artifact_id}_v{self.version}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Save manifest
        manifest = {
            'artifact_id': self.artifact_id,
            'version': self.version,
            'description': self.description,
            'creation_time': self.creation_time,
            'schema': self.schema,
            'fingerprint': self.fingerprint,
            'metadata': self.metadata,
            'files': {
                'manifest': 'manifest.json',
                'transform_state': 'transform_state.pkl',
                'schema': 'schema.json'
            }
        }
        
        with open(artifact_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        # Save schema separately for easy access
        with open(artifact_dir / 'schema.json', 'w') as f:
            json.dump(self.schema, f, indent=2, default=str)
        
        # Save transform state
        state_file = artifact_dir / 'transform_state.pkl'
        if compress:
            import gzip
            with gzip.open(str(state_file) + '.gz', 'wb') as f:
                pickle.dump(self.transform, f, protocol=pickle.HIGHEST_PROTOCOL)
            # Update manifest with compressed filename
            manifest['files']['transform_state'] = 'transform_state.pkl.gz'
            with open(artifact_dir / 'manifest.json', 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
        else:
            with open(state_file, 'wb') as f:
                pickle.dump(self.transform, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Artifact saved to {artifact_dir}")
        return artifact_dir
    
    @classmethod
    def load(cls, artifact_path: Union[str, Path], validate_schema: bool = True) -> 'TransformArtifact':
        """Load artifact from directory.
        
        Args:
            artifact_path: Path to artifact directory
            validate_schema: Whether to validate schema compatibility
            
        Returns:
            Loaded TransformArtifact
        """
        artifact_dir = Path(artifact_path)
        
        if not artifact_dir.exists():
            raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")
        
        # Load manifest
        manifest_path = artifact_dir / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Load transform state
        transform_state_file = manifest['files']['transform_state']
        state_path = artifact_dir / transform_state_file
        
        if transform_state_file.endswith('.gz'):
            import gzip
            with gzip.open(state_path, 'rb') as f:
                transform = pickle.load(f)
        else:
            with open(state_path, 'rb') as f:
                transform = pickle.load(f)
        
        # Validate schema if requested
        if validate_schema:
            current_fingerprint = cls._compute_current_fingerprint(transform)
            stored_fingerprint = manifest['fingerprint']
            
            if current_fingerprint != stored_fingerprint:
                logger.warning(f"Schema fingerprint mismatch for artifact {manifest['artifact_id']}")
                logger.warning(f"Stored: {stored_fingerprint}")
                logger.warning(f"Current: {current_fingerprint}")
        
        # Create artifact instance
        artifact = cls(
            transform=transform,
            artifact_id=manifest['artifact_id'],
            version=manifest['version'],
            description=manifest['description'],
            metadata=manifest.get('metadata', {})
        )
        
        # Restore saved metadata
        artifact.schema = manifest['schema']
        artifact.fingerprint = manifest['fingerprint']
        artifact.creation_time = manifest['creation_time']
        
        logger.info(f"Artifact loaded: {manifest['artifact_id']} v{manifest['version']}")
        return artifact
    
    @staticmethod
    def _compute_current_fingerprint(transform: Any) -> str:
        """Compute fingerprint for a transform (used for validation)."""
        # This is a simplified version of _generate_fingerprint
        # for validating loaded transforms
        fingerprint_data = {
            'transform_class': transform.__class__.__name__,
            'parameters': transform.get_params(),
        }
        
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()
    
    def validate_data_compatibility(self, data: Any) -> Dict[str, Any]:
        """Validate data compatibility with artifact schema.
        
        Args:
            data: Input data to validate
            
        Returns:
            Validation report dictionary
        """
        report = {
            'compatible': True,
            'warnings': [],
            'errors': [],
            'schema_changes': {}
        }
        
        # Check if we have fitted schema
        if 'fitted_schema' not in self.schema:
            report['warnings'].append("No fitted schema available for validation")
            return report
        
        fitted_schema = self.schema['fitted_schema']
        
        # Validate columns if available
        if hasattr(data, 'columns') and 'columns' in fitted_schema:
            expected_columns = set(fitted_schema['columns'])
            actual_columns = set(data.columns)
            
            missing_columns = expected_columns - actual_columns
            extra_columns = actual_columns - expected_columns
            
            if missing_columns:
                report['errors'].append(f"Missing columns: {list(missing_columns)}")
                report['compatible'] = False
            
            if extra_columns:
                report['warnings'].append(f"Extra columns: {list(extra_columns)}")
            
            report['schema_changes']['missing_columns'] = list(missing_columns)
            report['schema_changes']['extra_columns'] = list(extra_columns)
        
        # Validate data types if available
        if hasattr(data, 'dtypes') and 'dtypes' in fitted_schema:
            expected_dtypes = fitted_schema['dtypes']
            dtype_mismatches = []
            
            for col in expected_dtypes:
                if col in data.columns:
                    expected_dtype = expected_dtypes[col]
                    actual_dtype = str(data.dtypes[col])
                    
                    if expected_dtype != actual_dtype:
                        dtype_mismatches.append({
                            'column': col,
                            'expected': expected_dtype,
                            'actual': actual_dtype
                        })
            
            if dtype_mismatches:
                report['warnings'].extend([
                    f"Dtype mismatch in {m['column']}: expected {m['expected']}, got {m['actual']}"
                    for m in dtype_mismatches
                ])
                report['schema_changes']['dtype_mismatches'] = dtype_mismatches
        
        # Validate vocabulary for encoders
        if 'vocabularies' in self.schema and hasattr(self.transform, 'validate_categories'):
            for col, vocab in self.schema['vocabularies'].items():
                if hasattr(data, 'columns') and col in data.columns:
                    try:
                        col_report = self.transform.validate_categories(data, col)
                        
                        if col_report['unknown_count'] > 0:
                            report['warnings'].append(
                                f"Unknown categories in {col}: {col_report['unknown_categories']}"
                            )
                        
                        if col_report['stability_score'] < 0.8:
                            report['warnings'].append(
                                f"Low stability score for {col}: {col_report['stability_score']:.2f}"
                            )
                        
                    except Exception as e:
                        report['warnings'].append(f"Could not validate vocabulary for {col}: {e}")
        
        return report
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive artifact information.
        
        Returns:
            Dictionary with artifact details
        """
        return {
            'artifact_id': self.artifact_id,
            'version': self.version,
            'description': self.description,
            'creation_time': self.creation_time,
            'transform_class': self.transform.__class__.__name__,
            'transform_module': self.transform.__class__.__module__,
            'fingerprint': self.fingerprint,
            'schema': self.schema,
            'metadata': self.metadata,
            'is_fitted': getattr(self.transform, '_is_fitted', False)
        }


class ArtifactManager:
    """Manager for handling multiple transform artifacts."""
    
    def __init__(self, repository_path: Union[str, Path]):
        """Initialize artifact manager.
        
        Args:
            repository_path: Path to artifact repository directory
        """
        self.repository_path = Path(repository_path)
        self.repository_path.mkdir(parents=True, exist_ok=True)
        
        # Create index file if it doesn't exist
        self.index_path = self.repository_path / 'artifact_index.json'
        if not self.index_path.exists():
            self._create_empty_index()
        
        self.index = self._load_index()
    
    def _create_empty_index(self):
        """Create empty artifact index."""
        empty_index = {
            'artifacts': {},
            'created': datetime.datetime.utcnow().isoformat(),
            'last_updated': datetime.datetime.utcnow().isoformat()
        }
        
        with open(self.index_path, 'w') as f:
            json.dump(empty_index, f, indent=2)
    
    def _load_index(self) -> Dict[str, Any]:
        """Load artifact index."""
        with open(self.index_path, 'r') as f:
            return json.load(f)
    
    def _save_index(self):
        """Save artifact index."""
        self.index['last_updated'] = datetime.datetime.utcnow().isoformat()
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2, default=str)
    
    def store_artifact(self, artifact: TransformArtifact) -> Path:
        """Store artifact in repository.
        
        Args:
            artifact: TransformArtifact to store
            
        Returns:
            Path to stored artifact
        """
        # Save artifact to repository
        artifact_path = artifact.save(self.repository_path)
        
        # Update index
        artifact_info = {
            'artifact_id': artifact.artifact_id,
            'version': artifact.version,
            'description': artifact.description,
            'creation_time': artifact.creation_time,
            'transform_class': artifact.transform.__class__.__name__,
            'fingerprint': artifact.fingerprint,
            'path': str(artifact_path.relative_to(self.repository_path))
        }
        
        self.index['artifacts'][artifact.artifact_id] = artifact_info
        self._save_index()
        
        logger.info(f"Artifact {artifact.artifact_id} stored in repository")
        return artifact_path
    
    def load_artifact(self, artifact_id: str, validate_schema: bool = True) -> TransformArtifact:
        """Load artifact from repository.
        
        Args:
            artifact_id: Artifact ID to load
            validate_schema: Whether to validate schema
            
        Returns:
            Loaded TransformArtifact
        """
        if artifact_id not in self.index['artifacts']:
            raise KeyError(f"Artifact {artifact_id} not found in repository")
        
        artifact_info = self.index['artifacts'][artifact_id]
        artifact_path = self.repository_path / artifact_info['path']
        
        return TransformArtifact.load(artifact_path, validate_schema)
    
    def list_artifacts(self, transform_class: Optional[str] = None) -> List[Dict[str, Any]]:
        """List artifacts in repository.
        
        Args:
            transform_class: Filter by transform class name
            
        Returns:
            List of artifact information dictionaries
        """
        artifacts = list(self.index['artifacts'].values())
        
        if transform_class:
            artifacts = [a for a in artifacts if a['transform_class'] == transform_class]
        
        return artifacts
    
    def delete_artifact(self, artifact_id: str):
        """Delete artifact from repository.
        
        Args:
            artifact_id: Artifact ID to delete
        """
        if artifact_id not in self.index['artifacts']:
            raise KeyError(f"Artifact {artifact_id} not found in repository")
        
        # Delete artifact directory
        artifact_info = self.index['artifacts'][artifact_id]
        artifact_path = self.repository_path / artifact_info['path']
        
        if artifact_path.exists():
            import shutil
            shutil.rmtree(artifact_path)
        
        # Remove from index
        del self.index['artifacts'][artifact_id]
        self._save_index()
        
        logger.info(f"Artifact {artifact_id} deleted from repository")
    
    def cleanup_old_artifacts(self, keep_versions: int = 5):
        """Clean up old artifact versions, keeping most recent.
        
        Args:
            keep_versions: Number of versions to keep per transform class
        """
        # Group by transform class
        by_class = {}
        for artifact_id, info in self.index['artifacts'].items():
            transform_class = info['transform_class']
            if transform_class not in by_class:
                by_class[transform_class] = []
            by_class[transform_class].append((artifact_id, info))
        
        # Clean up each class
        for transform_class, artifacts in by_class.items():
            # Sort by creation time (newest first)
            artifacts.sort(key=lambda x: x[1]['creation_time'], reverse=True)
            
            # Delete old versions
            for artifact_id, info in artifacts[keep_versions:]:
                try:
                    self.delete_artifact(artifact_id)
                    logger.info(f"Cleaned up old artifact: {artifact_id}")
                except Exception as e:
                    logger.error(f"Failed to clean up artifact {artifact_id}: {e}")


# Convenience functions
def save_transform(transform: Any, directory: Union[str, Path], **kwargs) -> TransformArtifact:
    """Save a fitted transform as an artifact.
    
    Args:
        transform: Fitted transform to save
        directory: Directory to save in
        **kwargs: Additional arguments for TransformArtifact
        
    Returns:
        Created TransformArtifact
    """
    artifact = TransformArtifact(transform, **kwargs)
    artifact.save(directory)
    return artifact


def load_transform(artifact_path: Union[str, Path], validate_schema: bool = True) -> Any:
    """Load a transform from an artifact.
    
    Args:
        artifact_path: Path to artifact directory
        validate_schema: Whether to validate schema
        
    Returns:
        Loaded transform object
    """
    artifact = TransformArtifact.load(artifact_path, validate_schema)
    return artifact.transform