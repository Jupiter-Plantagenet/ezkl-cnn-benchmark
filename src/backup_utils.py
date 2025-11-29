"""
Backup utilities for experiment data and trained models.
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
import tarfile


def create_backup(source_dirs, backup_dir='./backups', backup_name=None):
    """
    Create a compressed backup of specified directories.
    
    Args:
        source_dirs: List of directories to backup
        backup_dir: Directory to store backups
        backup_name: Optional custom backup name
    
    Returns:
        Path to the backup file
    """
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    if backup_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"experiment_backup_{timestamp}"
    
    backup_path = backup_dir / f"{backup_name}.tar.gz"
    
    print(f"Creating backup: {backup_path}")
    
    with tarfile.open(backup_path, "w:gz") as tar:
        for source_dir in source_dirs:
            source_path = Path(source_dir)
            if source_path.exists():
                print(f"  Adding {source_path}...")
                tar.add(source_path, arcname=source_path.name)
    
    backup_size_mb = backup_path.stat().st_size / (1024 * 1024)
    print(f"✓ Backup created: {backup_path} ({backup_size_mb:.2f} MB)")
    
    return backup_path


def backup_trained_models(backup_dir='./backups'):
    """Backup trained model checkpoints"""
    return create_backup(
        source_dirs=['./models_pt'],
        backup_dir=backup_dir,
        backup_name=f"trained_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


def backup_results(backup_dir='./backups'):
    """Backup experiment results"""
    return create_backup(
        source_dirs=['./results'],
        backup_dir=backup_dir,
        backup_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


def backup_all(backup_dir='./backups'):
    """Backup both models and results"""
    return create_backup(
        source_dirs=['./models_pt', './results'],
        backup_dir=backup_dir,
        backup_name=f"full_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


def restore_backup(backup_path, target_dir='.'):
    """
    Restore from a backup file.
    
    Args:
        backup_path: Path to the backup tar.gz file
        target_dir: Directory to restore to
    """
    backup_path = Path(backup_path)
    target_dir = Path(target_dir)
    
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_path}")
    
    print(f"Restoring from: {backup_path}")
    
    with tarfile.open(backup_path, "r:gz") as tar:
        tar.extractall(path=target_dir)
    
    print(f"✓ Restore complete to {target_dir}")


def list_backups(backup_dir='./backups'):
    """List all available backups"""
    backup_dir = Path(backup_dir)
    
    if not backup_dir.exists():
        print("No backups directory found")
        return []
    
    backups = list(backup_dir.glob('*.tar.gz'))
    
    if not backups:
        print("No backups found")
        return []
    
    print(f"\nAvailable backups in {backup_dir}:")
    print("="*70)
    
    backup_info = []
    for backup in sorted(backups, reverse=True):
        stat = backup.stat()
        size_mb = stat.st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(stat.st_mtime)
        
        info = {
            'path': backup,
            'name': backup.name,
            'size_mb': size_mb,
            'modified': mtime
        }
        backup_info.append(info)
        
        print(f"{backup.name}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    return backup_info


def clean_stale_results(keep_backups=True):
    """
    Clean failed/stale experiment results while preserving trained models.
    
    Args:
        keep_backups: If True, create backup before cleaning
    """
    print("\n" + "="*70)
    print("CLEANING STALE RESULTS")
    print("="*70)
    
    # Backup current results if requested
    if keep_backups:
        print("\n1. Creating backup of current results...")
        try:
            backup_results()
        except Exception as e:
            print(f"Warning: Backup failed: {e}")
            response = input("Continue without backup? (y/N): ")
            if response.lower() != 'y':
                print("Cleanup cancelled")
                return
    
    # Clean results directories
    print("\n2. Removing stale experiment results...")
    results_dirs = [
        './results/core_layers',
        './results/scaling_study',
        './results/composite',
        './results/test_fix',
        './results/test_fix2',
        './results/test_final',
        './results/test_final2',
        './results/test_flat'
    ]
    
    for results_dir in results_dirs:
        path = Path(results_dir)
        if path.exists():
            try:
                shutil.rmtree(path)
                print(f"  ✓ Removed {path}")
            except Exception as e:
                print(f"  ✗ Failed to remove {path}: {e}")
    
    # Remove old experiment logs
    print("\n3. Removing old experiment logs...")
    results_path = Path('./results')
    if results_path.exists():
        for log_file in results_path.glob('experiment_log_*.json'):
            try:
                log_file.unlink()
                print(f"  ✓ Removed {log_file.name}")
            except Exception as e:
                print(f"  ✗ Failed to remove {log_file.name}: {e}")
    
    # Clean temp directories
    print("\n4. Cleaning temporary files...")
    temp_path = Path('./temp')
    if temp_path.exists():
        try:
            shutil.rmtree(temp_path)
            print(f"  ✓ Removed {temp_path}")
        except Exception as e:
            print(f"  ✗ Failed to remove {temp_path}: {e}")
    
    # Recreate clean directories
    print("\n5. Recreating clean directory structure...")
    for results_dir in ['./results/core_layers', './results/scaling_study', './results/composite']:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created {results_dir}")
    
    Path('./temp').mkdir(exist_ok=True)
    print(f"  ✓ Created ./temp")
    
    print("\n" + "="*70)
    print("✓ Cleanup complete!")
    print("  - Trained models preserved in ./models_pt/")
    if keep_backups:
        print("  - Results backed up in ./backups/")
    print("="*70 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Backup and restore utilities')
    parser.add_argument('action', choices=['backup', 'restore', 'list', 'clean'],
                        help='Action to perform')
    parser.add_argument('--models', action='store_true',
                        help='Backup models (for backup action)')
    parser.add_argument('--results', action='store_true',
                        help='Backup results (for backup action)')
    parser.add_argument('--all', action='store_true',
                        help='Backup everything (for backup action)')
    parser.add_argument('--file', type=str,
                        help='Backup file path (for restore action)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip backup when cleaning (for clean action)')
    
    args = parser.parse_args()
    
    if args.action == 'backup':
        if args.all:
            backup_all()
        elif args.models:
            backup_trained_models()
        elif args.results:
            backup_results()
        else:
            print("Please specify --models, --results, or --all")
    
    elif args.action == 'restore':
        if not args.file:
            print("Please specify --file <backup_file>")
        else:
            restore_backup(args.file)
    
    elif args.action == 'list':
        list_backups()
    
    elif args.action == 'clean':
        clean_stale_results(keep_backups=not args.no_backup)
