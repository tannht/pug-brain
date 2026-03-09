"""
Hatch build hook to build the React dashboard before packaging.

This ensures the dashboard is always included in the pip package.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class BuildDashboardHook(BuildHookInterface):
    """Hatch build hook to build dashboard before packaging."""

    PLUGIN_NAME = "build-dashboard"

    def initialize(
        self,
        version: str,  # noqa: ARG002
        build_data: dict,  # noqa: ARG002
    ) -> None:
        """Called before building the wheel/sdist."""
        # Get paths
        dashboard_dir = Path(self.root) / "dashboard"
        static_dist_dir = (
            Path(self.root)
            / "src"
            / "neural_memory"
            / "server"
            / "static"
            / "dist"
        )

        # Check if dashboard is already built
        index_html = static_dist_dir / "index.html"
        if index_html.exists():
            self.app.display_info("Dashboard already built, skipping...")
            return

        # Check if dashboard directory exists
        if not dashboard_dir.exists():
            self.app.display_warning("Dashboard directory not found, skipping build")
            return

        # Check if npm is available
        npm_path = shutil.which("npm")
        if not npm_path:
            self.app.display_warning(
                "npm not found, skipping dashboard build. "
                "Dashboard UI will not be available."
            )
            return

        # Check if node_modules exists
        node_modules = dashboard_dir / "node_modules"
        if not node_modules.exists():
            self.app.display_info("Installing dashboard dependencies...")
            try:
                subprocess.run(
                    [npm_path, "install"],
                    cwd=dashboard_dir,
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                self.app.display_error(f"Failed to install dashboard dependencies: {e}")
                return

        # Build the dashboard
        self.app.display_info("Building React dashboard...")
        try:
            subprocess.run(
                [npm_path, "run", "build"],
                cwd=dashboard_dir,
                check=True,
                capture_output=True,
            )
            self.app.display_success("Dashboard built successfully!")
        except subprocess.CalledProcessError as e:
            self.app.display_error(f"Failed to build dashboard: {e}")
            self.app.display_warning("Package will be built without dashboard UI")
