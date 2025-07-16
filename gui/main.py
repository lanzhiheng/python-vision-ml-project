#!/usr/bin/env python3
"""
Main GUI Application for ML/CV Project

This module provides a PyQt5-based GUI application for machine learning
and computer vision tasks.
"""

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                            QFileDialog, QGroupBox, QProgressBar, QTabWidget,
                            QGridLayout, QComboBox, QSpinBox, QCheckBox,
                            QMessageBox, QMenuBar, QAction, QStatusBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QFont, QIcon
import numpy as np
import cv2


class ImageProcessingWorker(QThread):
    """
    Worker thread for image processing tasks.
    """
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, image_path, operation):
        super().__init__()
        self.image_path = image_path
        self.operation = operation
    
    def run(self):
        try:
            # Simulate image processing
            for i in range(101):
                self.progress.emit(i)
                self.msleep(10)  # Simulate processing time
            
            self.finished.emit(f"Completed {self.operation} on {self.image_path}")
        except Exception as e:
            self.error.emit(str(e))


class MLCVMainWindow(QMainWindow):
    """
    Main window for the ML/CV GUI application.
    """
    
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.current_image = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("ML/CV Project - Machine Learning & Computer Vision Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create left panel (controls)
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Create right panel (display)
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
        
        # Apply styling
        self.apply_styling()
        
    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_action = QAction('Open Image', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        
        save_action = QAction('Save Result', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_result)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        cv_tools_action = QAction('Computer Vision Tools', self)
        cv_tools_action.triggered.connect(self.show_cv_tools)
        tools_menu.addAction(cv_tools_action)
        
        ml_tools_action = QAction('Machine Learning Tools', self)
        ml_tools_action.triggered.connect(self.show_ml_tools)
        tools_menu.addAction(ml_tools_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_left_panel(self):
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File operations group
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        
        self.open_btn = QPushButton("Open Image")
        self.open_btn.clicked.connect(self.open_image)
        file_layout.addWidget(self.open_btn)
        
        self.save_btn = QPushButton("Save Result")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        file_layout.addWidget(self.save_btn)
        
        layout.addWidget(file_group)
        
        # Image processing group
        processing_group = QGroupBox("Image Processing")
        processing_layout = QVBoxLayout(processing_group)
        
        self.blur_btn = QPushButton("Apply Blur")
        self.blur_btn.clicked.connect(lambda: self.process_image("blur"))
        self.blur_btn.setEnabled(False)
        processing_layout.addWidget(self.blur_btn)
        
        self.edges_btn = QPushButton("Detect Edges")
        self.edges_btn.clicked.connect(lambda: self.process_image("edges"))
        self.edges_btn.setEnabled(False)
        processing_layout.addWidget(self.edges_btn)
        
        self.grayscale_btn = QPushButton("Convert to Grayscale")
        self.grayscale_btn.clicked.connect(lambda: self.process_image("grayscale"))
        self.grayscale_btn.setEnabled(False)
        processing_layout.addWidget(self.grayscale_btn)
        
        layout.addWidget(processing_group)
        
        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QGridLayout(params_group)
        
        params_layout.addWidget(QLabel("Blur Kernel Size:"), 0, 0)
        self.blur_size = QSpinBox()
        self.blur_size.setRange(1, 31)
        self.blur_size.setValue(5)
        self.blur_size.setSingleStep(2)
        params_layout.addWidget(self.blur_size, 0, 1)
        
        params_layout.addWidget(QLabel("Edge Threshold:"), 1, 0)
        self.edge_threshold = QSpinBox()
        self.edge_threshold.setRange(1, 255)
        self.edge_threshold.setValue(100)
        params_layout.addWidget(self.edge_threshold, 1, 1)
        
        layout.addWidget(params_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
        
    def create_right_panel(self):
        """Create the right display panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget for different views
        self.tab_widget = QTabWidget()
        
        # Image display tab
        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #ccc; background-color: #f9f9f9;")
        self.image_label.setMinimumSize(400, 300)
        image_layout.addWidget(self.image_label)
        
        self.tab_widget.addTab(image_tab, "Image Display")
        
        # Log tab
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        self.tab_widget.addTab(log_tab, "Processing Log")
        
        layout.addWidget(self.tab_widget)
        
        return panel
        
    def apply_styling(self):
        """Apply custom styling to the application."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QLabel {
                font-size: 12px;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
                color: white;
            }
        """)
        
    def open_image(self):
        """Open an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        
        if file_path:
            try:
                # Load image with OpenCV
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    raise ValueError("Could not load image")
                
                self.current_image_path = file_path
                self.display_image(self.current_image)
                self.enable_processing_buttons(True)
                self.log_message(f"Loaded image: {os.path.basename(file_path)}")
                self.status_bar.showMessage(f"Image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load image: {str(e)}")
                
    def display_image(self, image):
        """Display an image in the GUI."""
        try:
            # Convert BGR to RGB for display
            if len(image.shape) == 3:
                display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                display_image = image
            
            # Convert to QPixmap
            height, width = display_image.shape[:2]
            if len(display_image.shape) == 3:
                bytes_per_line = 3 * width
                q_image = QPixmap.fromImage(
                    QPixmap.fromImage(
                        QPixmap.fromRawData(
                            display_image.data, width, height, 
                            bytes_per_line, QPixmap.Format_RGB888
                        ).toImage()
                    )
                )
            else:
                bytes_per_line = width
                q_image = QPixmap.fromImage(
                    QPixmap.fromRawData(
                        display_image.data, width, height, 
                        bytes_per_line, QPixmap.Format_Grayscale8
                    ).toImage()
                )
            
            # Scale image to fit label
            scaled_pixmap = q_image.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.log_message(f"Error displaying image: {str(e)}")
            
    def process_image(self, operation):
        """Process the current image."""
        if self.current_image is None:
            return
            
        try:
            result_image = None
            
            if operation == "blur":
                kernel_size = self.blur_size.value()
                if kernel_size % 2 == 0:  # Ensure odd kernel size
                    kernel_size += 1
                result_image = cv2.GaussianBlur(self.current_image, (kernel_size, kernel_size), 0)
                self.log_message(f"Applied Gaussian blur with kernel size {kernel_size}")
                
            elif operation == "edges":
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
                threshold = self.edge_threshold.value()
                result_image = cv2.Canny(gray, threshold, threshold * 2)
                self.log_message(f"Detected edges with threshold {threshold}")
                
            elif operation == "grayscale":
                result_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
                self.log_message("Converted to grayscale")
            
            if result_image is not None:
                self.current_image = result_image
                self.display_image(result_image)
                self.save_btn.setEnabled(True)
                self.status_bar.showMessage(f"Applied {operation}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing image: {str(e)}")
            self.log_message(f"Error: {str(e)}")
            
    def save_result(self):
        """Save the processed image."""
        if self.current_image is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", 
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if file_path:
            try:
                success = cv2.imwrite(file_path, self.current_image)
                if success:
                    self.log_message(f"Saved image to: {os.path.basename(file_path)}")
                    self.status_bar.showMessage(f"Image saved: {os.path.basename(file_path)}")
                    QMessageBox.information(self, "Success", "Image saved successfully!")
                else:
                    raise ValueError("Failed to save image")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save image: {str(e)}")
                
    def enable_processing_buttons(self, enabled):
        """Enable or disable processing buttons."""
        self.blur_btn.setEnabled(enabled)
        self.edges_btn.setEnabled(enabled)
        self.grayscale_btn.setEnabled(enabled)
        
    def log_message(self, message):
        """Add a message to the log."""
        self.log_text.append(f"[{QTimer().remainingTime()}] {message}")
        
    def show_cv_tools(self):
        """Show computer vision tools dialog."""
        QMessageBox.information(self, "Computer Vision Tools", 
                               "Advanced CV tools will be available in future versions.")
        
    def show_ml_tools(self):
        """Show machine learning tools dialog."""
        QMessageBox.information(self, "Machine Learning Tools", 
                               "ML model training tools will be available in future versions.")
        
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About ML/CV Project", 
                         "Machine Learning & Computer Vision Project\\n"
                         "Version 0.1.0\\n\\n"
                         "Built with PyTorch, OpenCV, and PyQt5\\n"
                         "For educational and research purposes.")


class MLCVApp:
    """
    Main application class.
    """
    
    def __init__(self):
        self.app = None
        self.window = None
        
    def run(self):
        """Run the GUI application."""
        try:
            # Create QApplication
            self.app = QApplication(sys.argv)
            self.app.setApplicationName("ML/CV Project")
            self.app.setApplicationVersion("0.1.0")
            
            # Create main window
            self.window = MLCVMainWindow()
            self.window.show()
            
            # Start event loop
            sys.exit(self.app.exec_())
            
        except Exception as e:
            print(f"Error running application: {e}")
            if self.app:
                self.app.quit()


def demo_gui():
    """
    Demonstrate GUI functionality without starting the full application.
    """
    print("=== PyQt5 GUI Demo ===")
    print("GUI components available:")
    print("  - Main window with menu bar and status bar")
    print("  - File operations (open/save)")
    print("  - Image processing controls")
    print("  - Real-time image display")
    print("  - Processing log")
    print("  - Parameter controls")
    print("\\nTo run the full GUI application, use: python gui/main.py")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_gui()
    else:
        app = MLCVApp()
        app.run()