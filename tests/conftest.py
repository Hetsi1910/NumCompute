"""Shared pytest configuration: headless matplotlib backend."""
import matplotlib

matplotlib.use("Agg")
