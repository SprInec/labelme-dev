"""
主题样式定义模块，提供明亮和暗黑两套主题样式。
"""
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

# 明亮主题样式表
LIGHT_STYLE = """
/* 全局字体设置 */
* {
    font-family: "Microsoft YaHei UI", "Segoe UI", "Arial", sans-serif;
    font-size: 10pt;
    color: #333333;
    letter-spacing: 0.4px;
    font-weight: 400;
}


QMainWindow, QDialog {
    background-color: #ffffff;
}

QMenuBar {
    background-color: #f8f9fa;
    color: #333333;
    border-bottom: 1px solid #e6e6e6;
    padding: 1px;
    font-size: 10pt;
}

QMenuBar::item {
    padding: 3px 6px;
    border-radius: 4px;
    margin: 1px 1px;
}

QMenuBar::item:selected {
    background-color: #e8f0fe;
    color: #1967d2;
}

QMenuBar::item:pressed {
    background-color: #d2e3fc;
}

QMenu {
    background-color: #ffffff;
    color: #333333;
    border: 1px solid #e6e6e6;
    border-radius: 8px;
    padding: 8px 0;
}

QMenu::item {
    padding: 8px 30px 8px 20px;
    border-radius: 4px;
    margin: 2px 4px;
}

QMenu::item:selected {
    background-color: #e8f0fe;
    color: #1967d2;
}

QMenu::separator {
    height: 1px;
    background-color: #e6e6e6;
    margin: 6px 0;
}

/* 菜单中的复选框优化样式 */
QMenu::indicator {
    width: 20px;
    height: 20px;
    margin-left: 6px;
    subcontrol-position: left center;
    subcontrol-origin: padding;
    position: absolute;
    left: 4px;
}

QMenu::indicator:non-exclusive:unchecked {
    border: 2px solid #bdbdbd;
    border-radius: 3px;
    background-color: #ffffff;
}

QMenu::indicator:non-exclusive:checked {
    border: 2px solid #4285f4;
    border-radius: 3px;
    background-color: #4285f4;
    image: url(:/check-white.png);
}

QMenu::indicator:exclusive:unchecked {
    border: 2px solid #bdbdbd;
    border-radius: 10px;
    background-color: #ffffff;
}

QMenu::indicator:exclusive:checked {
    border: 2px solid #4285f4;
    border-radius: 10px;
    background-color: #ffffff;
    image: url(:/dot-blue.png);
}

/* 工具栏样式优化 */
QToolBar {
    background-color: #f8f9fa;
    border-bottom: 1px solid #e6e6e6;
    spacing: 8px;
    padding: 10px 6px;
    min-width: 30px;
}

QToolBar::separator {
    width: 1px;
    background-color: #e6e6e6;
    margin: 8px 4px;
}

QToolButton {
    padding: 6px 10px;
    border-radius: 8px;
    font-size: 8.5pt;
    color: #333333;
    background-color: transparent;
    margin: 3px;
    min-width: 35px;
    min-height: 35px;
    text-align: center;
}

QToolButton[popupMode="1"] {
    padding-right: 6px;
}

QToolButton::menu-button {
    border: none;
    width: 18px;
}

QToolButton::menu-arrow {
    image: url(:/down-arrow.png);
    width: 12px;
    height: 12px;
}

QToolButton:hover {
    background-color: #f5f5f5;
}

QToolButton:pressed {
    background-color: #e8f0fe;
}

QToolButton:checked {
    background-color: #e8f0fe;
    color: #1967d2;
}

QToolBar::handle {
    background-color: #e6e6e6;
    width: 6px;
    height: 6px;
    border-radius: 3px;
    margin: 2px;
}

QStatusBar {
    background-color: #f8f9fa;
    color: #333333;
    border-top: 1px solid #e6e6e6;
    padding: 3px;
    font-size: 9pt;
}

QListWidget, QTreeView, QListView, #labelListWidget, #polygonListWidget, #labelListContainer, #polygonListContainer, #flagWidgetContainer {
    background-color: #ffffff;
    alternate-background-color: #f8f9fa;
    border: 1px solid #e6e6e6;
    border-radius: 8px;
    padding: 5px;
    font-size: 9pt;
}

QListWidget::item, QTreeView::item, QListView::item, #labelListWidget::item, #polygonListWidget::item {
    padding: 7px;
    border-radius: 5px;
    font-size: 9.5pt;
    margin: 2px 1px;
    color: #333333;
}

QListWidget::item:hover, QTreeView::item:hover, QListView::item:hover {
    background-color: #f5f5f5;
}

QListWidget::item:selected, QTreeView::item:selected, QListView::item:selected {
    background-color: #e8f0fe;
    color: #1967d2;
}

/* 优化树状视图样式 */
QTreeView {
    show-decoration-selected: 1;
    outline: none;
}

QTreeView::item {
    height: 30px;  /* 增加高度提供更多空间 */
    color: #333333;
    padding: 4px 6px;
    margin: 2px 0px;
    border-radius: 5px;
}

QTreeView::item:hover {
    background-color: #f5f5f5;
}

QTreeView::item:selected {
    background-color: #e8f0fe;
    color: #1967d2;
}

QTreeView::branch {
    background-color: transparent;
    padding-left: 2px;
}

QTreeView::branch:has-children:!has-siblings:closed,
QTreeView::branch:closed:has-children:has-siblings {
    image: url(:/right-arrow.png);
    padding-left: 5px;
}

QTreeView::branch:open:has-children:!has-siblings,
QTreeView::branch:open:has-children:has-siblings {
    image: url(:/down-arrow.png);
    padding-left: 5px;
}

/* 自定义标签列表和多边形标签的样式 */
#labelListWidget, #polygonListWidget {
    padding: 12px;
    margin: 8px 0;
    border-radius: 10px;
    background-color: #f8f9fa;
    border: none;
}

#labelListWidget QStandardItem, #polygonListWidget QStandardItem {
    height: 36px;
    padding: 4px 10px;
    margin: 3px 0px;
    border-radius: 6px;
}

#labelListWidget QLabel, #polygonListWidget QLabel {
    font-size: 10pt;
    color: #333333;
    font-weight: 400;
    letter-spacing: 0.2px;
}

#labelListContainer, #polygonListContainer, #flagWidgetContainer {
    background-color: #ffffff;
    border: none;
    border-radius: 12px;
    padding: 10px;
}

/* 统一所有复选框样式 */
QCheckBox, QTreeView::indicator, QListView::indicator, #labelListWidget::indicator, #polygonListWidget::indicator {
    spacing: 10px;
}

QCheckBox::indicator, QTreeView::indicator, QListView::indicator, #labelListWidget::indicator, #polygonListWidget::indicator {
    width: 22px;  /* 增加复选框尺寸 */
    height: 22px;
    border: 2px solid #bdbdbd;
    border-radius: 4px;
    background-color: #ffffff;
}

QCheckBox::indicator:checked, QTreeView::indicator:checked, QListView::indicator:checked, #labelListWidget::indicator:checked, #polygonListWidget::indicator:checked {
    background-color: #4285f4;
    border: 2px solid #4285f4;
    image: url(:/check-white.png);
}

QCheckBox::indicator:unchecked:hover, QTreeView::indicator:unchecked:hover, QListView::indicator:unchecked:hover, #labelListWidget::indicator:unchecked:hover, #polygonListWidget::indicator:unchecked:hover {
    border: 2px solid #4285f4;
}

QDockWidget {
    border: 1px solid #e6e6e6;
    border-radius: 8px;
    titlebar-close-icon: url(:/close.png);
}

QDockWidget::title {
    background-color: #f8f9fa;
    padding: 8px;
    text-align: left;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}

QPushButton {
    background-color: #4285f4;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #5094ed;
}

QPushButton:pressed {
    background-color: #3367d6;
}

QPushButton:disabled {
    background-color: #e0e0e0;
    color: #9e9e9e;
}

QLabel {
    color: #333333;
}

QLineEdit, QComboBox {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 5px;
    color: #333333;
}

QLineEdit:focus, QComboBox:focus {
    border: 1px solid #2196f3;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox::down-arrow {
    image: url(:/down-arrow.png);
    width: 12px;
    height: 12px;
}

QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    selection-background-color: #e3f2fd;
    selection-color: #1976d2;
    color: #333333;
}

QScrollBar:vertical {
    background-color: #f5f5f5;
    width: 14px;
    margin: 0px;
    border-radius: 7px;
}

QScrollBar::handle:vertical {
    background-color: #bdbdbd;
    min-height: 30px;
    border-radius: 7px;
}

QScrollBar::handle:vertical:hover {
    background-color: #9e9e9e;
}

QScrollBar:horizontal {
    background-color: #f5f5f5;
    height: 14px;
    margin: 0px;
    border-radius: 7px;
}

QScrollBar::handle:horizontal {
    background-color: #bdbdbd;
    min-width: 30px;
    border-radius: 7px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #9e9e9e;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

QTabWidget::pane {
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    top: -1px;
}

QTabBar::tab {
    background-color: #f5f5f5;
    border: 1px solid #e0e0e0;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 6px 12px;
    margin-right: 2px;
    font-weight: bold;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    border-bottom-color: #ffffff;
}

QTabBar::tab:hover:!selected {
    background-color: #e3f2fd;
}

QHeaderView::section {
    background-color: #f5f5f5;
    padding: 5px;
    border: 1px solid #e0e0e0;
    font-weight: bold;
}

QToolTip {
    background-color: #ffffff;
    color: #2c3e50;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 4px;
}

QRadioButton {
    spacing: 8px;
}

QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #bdbdbd;
    border-radius: 9px;
}

QRadioButton::indicator:checked {
    background-color: #2196f3;
    border: 1px solid #2196f3;
    width: 10px;
    height: 10px;
    border-radius: 5px;
}

QRadioButton::indicator:unchecked:hover {
    border: 1px solid #2196f3;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #e0e0e0;
    margin: 0px;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #4285f4;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #1967d2;
}

QProgressBar {
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    background-color: #f5f5f5;
    text-align: center;
    height: 12px;
}

QProgressBar::chunk {
    background-color: #4285f4;
    border-radius: 4px;
}

QSplitter::handle {
    background-color: #e0e0e0;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

/* 现代化组件样式 */
QGroupBox {
    border: 1px solid #e6e6e6;
    border-radius: 4px;
    margin-top: 12px;
    padding: 12px;
    background-color: #ffffff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    background-color: #ffffff;
    font-weight: 500;
    color: #1967d2;
}

QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit, QDateTimeEdit {
    background-color: #ffffff;
    border: 1px solid #e6e6e6;
    border-radius: 4px;
    padding: 5px;
    min-height: 24px;
}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QDateEdit::up-button, QTimeEdit::up-button, QDateTimeEdit::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #e6e6e6;
    border-bottom: 1px solid #e6e6e6;
    border-top-right-radius: 4px;
    background-color: #f8f9fa;
}

QSpinBox::down-button, QDoubleSpinBox::down-button,
QDateEdit::down-button, QTimeEdit::down-button, QDateTimeEdit::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    border-left: 1px solid #e6e6e6;
    border-top-right-radius: 0;
    border-bottom-right-radius: 4px;
    background-color: #f8f9fa;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QDateEdit::up-button:hover, QTimeEdit::up-button:hover, QDateTimeEdit::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover, 
QDateEdit::down-button:hover, QTimeEdit::down-button:hover, QDateTimeEdit::down-button:hover {
    background-color: #e8f0fe;
}

/* 美化下拉菜单 */
QComboBox {
    min-height: 24px;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 24px;
    border-left: none;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
}

/* 美化标签和按钮 */
QLabel {
    color: #333333;
    padding: 2px;
}

QPushButton {
    min-height: 24px;
    font-weight: 500;
}

/* 美化对话框和窗口 */
QDialog {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 20px;  /* 增加内边距 */
}

QDialog QLabel {
    color: #333333;
    font-size: 10pt;
    margin-bottom: 8px;
    font-weight: 500;
}

QDialog QLineEdit {
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 10pt;
    margin-bottom: 12px;
}

QDialog QListWidget {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 15px;
    font-size: 10pt;
}

QDialog QPushButton {
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 10pt;
}

QDialog QPushButton:default {
    background-color: #1967d2;
    color: white;
    border: none;
}

QDialog QPushButton:default:hover {
    background-color: #1a73e8;
}

QDialog QPushButton:default:pressed {
    background-color: #1557b0;
}

QDialog QPushButton:!default {
    background-color: #f8f9fa;
    color: #333333;
    border: 1px solid #dadce0;
}

QDialog QPushButton:!default:hover {
    background-color: #e8f0fe;
    border-color: #1967d2;
}

/* 美化滑块 */
QSlider::groove:horizontal {
    height: 4px;
    background: #e6e6e6;
    margin: 0px;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #4285f4;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #1967d2;
}

QWidget {
    color: #f0f0f0;  /* 修改为亮色而不是黑色 */
}

/* 多边形标签列表样式 */
#polygonListWidget::item, #labelListWidget::item {
    padding: 8px;
    border-radius: 6px;
    margin: 2px 0px;
    min-height: 24px;
    font-size: 10pt;
    color: #333333;
}

#polygonListWidget::item:hover, #labelListWidget::item:hover {
    background-color: #f5f5f5;
}

#polygonListWidget::item:selected, #labelListWidget::item:selected {
    background-color: #e8f0fe;
    color: #1967d2;
}

/* 确保标签对话框中的控件垂直居中 */
QLabel {
    margin-top: 0px;
    margin-bottom: 0px;
}

/* 优化标签对话框中的按钮布局 */
QDialogButtonBox {
    spacing: 10px;
    margin-top: 10px;
}

QDialogButtonBox QPushButton {
    min-width: 80px;
    padding: 8px 16px;
}

/* 优化标签列表样式 */
QListWidget {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    outline: none;
    background-color: #ffffff;
}

QListWidget::item {
    border-radius: 6px;
    padding: 8px;
    margin: 2px 0px;
    min-height: 24px;
}

QListWidget::item:hover {
    background-color: #f5f5f5;
}

QListWidget::item:selected {
    background-color: #e8f0fe;
    color: #1967d2;
}

QDialog {
    background-color: #ffffff;
    border-radius: 10px;
}
QLineEdit {
    padding: 8px 12px;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    background-color: #f8f9fa;
    margin-bottom: 10px;
    font-size: 10pt;
}
QLineEdit:focus {
    border-color: #4285f4;
    background-color: #ffffff;
}
QListWidget {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 5px;
    background-color: #ffffff;
}
QListWidget::item {
    padding: 8px 12px;
    border-radius: 6px;
    margin: 2px 1px;
}
QListWidget::item:hover {
    background-color: #f5f5f5;
}
QListWidget::item:selected {
    background-color: #e8f0fe;
    color: #1967d2;
}
QPushButton {
    padding: 8px 20px;
    border-radius: 6px;
    font-size: 8pt;
    font-weight: 500;
    margin: 5px;
    min-width: 90px;
}
QPushButton#color_button {
    padding: 0px;
    min-width: 28px;
    border-radius: 14px;
    border: 1px solid #e0e0e0;
    margin-top: 1px;
}

/* 确保整个应用程序使用正确的文字颜色 */
QMainWindow, QDialog, QWidget, QLabel, QPushButton, QLineEdit, QTreeView, QListView, QMenu, QMenuBar {
    color: #333333;
}
"""

# 暗黑主题样式表
DARK_STYLE = """
/* 全局字体设置 */
* {
    font-family: "Microsoft YaHei UI", "Segoe UI", "Arial", sans-serif;
    font-size: 10pt;
    color: #e0e0e0;
    letter-spacing: 0.4px;
    font-weight: 400;
}

QMainWindow, QDialog {
    background-color: #1e1e1e;
}

QMenuBar {
    background-color: #2d2d30;
    color: #cccccc;
    border-bottom: 1px solid #3e3e42;
    padding: 1px;
    font-size: 10pt;
}

QMenuBar::item {
    padding: 3px 6px;
    border-radius: 4px;
    margin: 1px 1px;
}

QMenuBar::item:selected {
    background-color: #3e3e42;
    color: #ffffff;
}

QMenuBar::item:pressed {
    background-color: #007acc;
}

QMenu {
    background-color: #2d2d30;
    color: #cccccc;
    border: 1px solid #3e3e42;
    border-radius: 8px;
    padding: 8px 0;
}

QMenu::item {
    padding: 8px 30px 8px 20px;
    border-radius: 4px;
    margin: 2px 4px;
}

QMenu::item:selected {
    background-color: #3e3e42;
    color: #ffffff;
}

QMenu::separator {
    height: 1px;
    background-color: #3e3e42;
    margin: 6px 0;
}

/* 暗色主题菜单中的复选框统一样式 */
QMenu::indicator {
    width: 20px;
    height: 20px;
    margin-left: 6px;
    subcontrol-position: left center;
    subcontrol-origin: padding;
    position: absolute;
    left: 4px;
}

QMenu::indicator:non-exclusive:unchecked {
    border: 2px solid rgb(58, 57, 57);
    border-radius: 3px;
    background-color: #2d2d30;
}

QMenu::indicator:non-exclusive:checked {
    border: 2px solid #007acc;
    border-radius: 3px;
    background-color: #007acc;
    image: url(:/check-white.png);
}

QMenu::indicator:exclusive:unchecked {
    border: 2px solid rgb(50, 49, 49);
    border-radius: 10px;
    background-color: #2d2d30;
}

QMenu::indicator:exclusive:checked {
    border: 2px solid #007acc;
    border-radius: 10px;
    background-color: #2d2d30;
    image: url(:/dot-blue-dark.png);
}

QToolBar {
    background-color: #2d2d30;
    border-bottom: 1px solid #3e3e42;
    spacing: 8px;
    padding: 10px 6px;
    min-width: 30px;
}

QToolBar::separator {
    width: 1px;
    background-color: #3e3e42;
    margin: 8px 4px;
}

QToolButton {
    padding: 6px 10px;
    border-radius: 8px;
    font-size: 8.5pt;
    color: #cccccc;
    background-color: transparent;
    margin: 3px;
    min-width: 35px;
    min-height: 35px;
    text-align: center;
}

QToolButton[popupMode="1"] {
    padding-right: 6px;
}

QToolButton::menu-button {
    border: none;
    width: 18px;
}

QToolButton::menu-arrow {
    image: url(:/down-arrow-dark.png);
    width: 12px;
    height: 12px;
}

QToolButton:hover {
    background-color: #3e3e42;
}

QToolButton:pressed {
    background-color: #007acc;
}

QToolButton:checked {
    background-color: #007acc;
    border: 1px solid #1c97ea;
}

QToolBar::handle {
    background-color: #3e3e42;
    width: 6px;
    height: 6px;
    border-radius: 3px;
    margin: 2px;
}

QStatusBar {
    background-color: #2d2d30;
    color: #cccccc;
    border-top: 1px solid #3e3e42;
    padding: 3px;
    font-size: 9pt;
}

QListWidget, QTreeView, QListView, #labelListWidget, #polygonListWidget, #labelListContainer, #polygonListContainer, #flagWidgetContainer {
    background-color: #252526;
    alternate-background-color: #2d2d30;
    border: 1px solid #3e3e42;
    border-radius: 8px;
    padding: 5px;
    color: #cccccc;
    font-size: 9pt;
}

QListWidget::item, QTreeView::item, QListView::item, #labelListWidget::item, #polygonListWidget::item {
    padding: 7px;
    border-radius: 5px;
    font-size: 9.5pt;
    margin: 2px 1px;
}

QListWidget::item:hover, QTreeView::item:hover, QListView::item:hover {
    background-color: #3e3e42;
}

QListWidget::item:selected, QTreeView::item:selected, QListView::item:selected {
    background-color: #007acc;
    color: #ffffff;
}

/* 暗色主题树状视图样式 */
QTreeView {
    show-decoration-selected: 1;
    outline: none;
}

QTreeView::item {
    height: 30px;  /* 增加高度提供更多空间 */
    color: #cccccc;
    padding: 4px 6px;
    margin: 2px 0px;
    border-radius: 5px;
}

QTreeView::item:hover {
    background-color: #3e3e42;
}

QTreeView::item:selected {
    background-color: #007acc;
    color: #ffffff;
}

QTreeView::branch {
    background-color: transparent;
    padding-left: 2px;
}

QTreeView::branch:has-children:!has-siblings:closed,
QTreeView::branch:closed:has-children:has-siblings {
    image: url(:/right-arrow-dark.png);
    padding-left: 5px;
}

QTreeView::branch:open:has-children:!has-siblings,
QTreeView::branch:open:has-children:has-siblings {
    image: url(:/down-arrow-dark.png);
    padding-left: 5px;
}

/* 自定义标签列表和多边形标签的样式 */
#labelListWidget, #polygonListWidget {
    padding: 12px;
    margin: 8px 0;
    border-radius: 10px;
    background-color: #2d2d30;
    border: none;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}

#labelListWidget QStandardItem, #polygonListWidget QStandardItem {
    height: 36px;
    padding: 4px 10px;
    margin: 3px 0px;
    border-radius: 6px;
}

#labelListWidget QLabel, #polygonListWidget QLabel {
    font-size: 10pt;
    color: #e0e0e0;
    font-weight: 400;
    letter-spacing: 0.2px;
}

#labelListContainer, #polygonListContainer, #flagWidgetContainer {
    background-color: #252526;
    border: none;
    border-radius: 12px;
    padding: 10px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
}

/* 统一所有复选框样式 */
QCheckBox, QTreeView::indicator, QListView::indicator, #labelListWidget::indicator, #polygonListWidget::indicator {
    spacing: 8px;
    color: #cccccc;
}

QCheckBox::indicator, QTreeView::indicator, QListView::indicator, #labelListWidget::indicator, #polygonListWidget::indicator {
    width: 20px;
    height: 20px;
    border: 2px solid #5d5d5d;
    border-radius: 4px;
    background-color: #ffffff;
}

QCheckBox::indicator:checked, QTreeView::indicator:checked, QListView::indicator:checked, #labelListWidget::indicator:checked, #polygonListWidget::indicator:checked {
    background-color: #007acc;
    border: 2px solid #007acc;
    image: url(:/check-white.png);
}

QCheckBox::indicator:unchecked:hover, QTreeView::indicator:unchecked:hover, QListView::indicator:unchecked:hover, #labelListWidget::indicator:unchecked:hover, #polygonListWidget::indicator:unchecked:hover {
    border: 2px solid #007acc;
}

QDockWidget {
    border: 1px solid #3e3e42;
    border-radius: 8px;
    titlebar-close-icon: url(:/close-dark.png);
    color: #e0e0e0;
}

QDockWidget::title {
    background-color: #2d2d30;
    padding: 8px;
    text-align: left;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}

QPushButton {
    background-color: #007acc;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #1c97ea;
}

QPushButton:pressed {
    background-color: #0062a3;
}

QPushButton:disabled {
    background-color: #3e3e42;
    color: #777777;
}

QLabel {
    color: #cccccc;
}

QLineEdit, QComboBox {
    background-color: #3c3c3c;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 5px;
    color: #cccccc;
}

QLineEdit:focus, QComboBox:focus {
    border: 1px solid #007acc;
    background-color: #3c3c3c;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox::down-arrow {
    image: url(:/down-arrow-dark.png);
    width: 12px;
    height: 12px;
}

QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    selection-background-color: #094771;
    selection-color: #ffffff;
}

QScrollBar:vertical {
    background-color: #1e1e1e;
    width: 14px;
    margin: 0px;
    border-radius: 7px;
}

QScrollBar::handle:vertical {
    background-color: #424242;
    min-height: 30px;
    border-radius: 7px;
}

QScrollBar::handle:vertical:hover {
    background-color: #5d5d5d;
}

QScrollBar:horizontal {
    background-color: #252525;
    height: 14px;
    margin: 0px;
    border-radius: 7px;
}

QScrollBar::handle:horizontal {
    background-color: #4d4d4d;
    min-width: 30px;
    border-radius: 7px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #5d5d5d;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

QTabWidget::pane {
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    top: -1px;
}

QTabBar::tab {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 6px 12px;
    margin-right: 2px;
    font-weight: bold;
    color: #e0e0e0;
}

QTabBar::tab:selected {
    background-color: #3d3d3d;
    border-bottom-color: #3d3d3d;
}

QTabBar::tab:hover:!selected {
    background-color: #333333;
}

QHeaderView::section {
    background-color: #2d2d2d;
    padding: 5px;
    border: 1px solid #3d3d3d;
    color: #e0e0e0;
    font-weight: bold;
}

QToolTip {
    background-color: #2d2d2d;
    color: #e0e0e0;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 4px;
}

QRadioButton {
    spacing: 8px;
    color: #e0e0e0;
}

QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #5d5d5d;
    border-radius: 9px;
}

QRadioButton::indicator:checked {
    background-color: #64b5f6;
    border: 1px solid #64b5f6;
    width: 10px;
    height: 10px;
    border-radius: 5px;
}

QRadioButton::indicator:unchecked:hover {
    border: 1px solid #64b5f6;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #3d3d3d;
    margin: 0px;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #64b5f6;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #42a5f5;
}

QProgressBar {
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    background-color: #252525;
    text-align: center;
    height: 12px;
}

QProgressBar::chunk {
    background-color: #64b5f6;
    border-radius: 4px;
}

QSplitter::handle {
    background-color: #3d3d3d;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

/* 暗色主题现代化组件样式 */
QGroupBox {
    border: 1px solid #3e3e42;
    border-radius: 4px;
    margin-top: 12px;
    padding: 12px;
    background-color: #252526;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    background-color: #252526;
    font-weight: 500;
    color: #007acc;
}

QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit, QDateTimeEdit {
    background-color: #2d2d30;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 5px;
    min-height: 24px;
    color: #cccccc;
}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QDateEdit::up-button, QTimeEdit::up-button, QDateTimeEdit::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #3e3e42;
    border-bottom: 1px solid #3e3e42;
    border-top-right-radius: 4px;
    background-color: #2d2d30;
}

QSpinBox::down-button, QDoubleSpinBox::down-button,
QDateEdit::down-button, QTimeEdit::down-button, QDateTimeEdit::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    border-left: 1px solid #3e3e42;
    border-top-right-radius: 0;
    border-bottom-right-radius: 4px;
    background-color: #2d2d30;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QDateEdit::up-button:hover, QTimeEdit::up-button:hover, QDateTimeEdit::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover, 
QDateEdit::down-button:hover, QTimeEdit::down-button:hover, QDateTimeEdit::down-button:hover {
    background-color: #3e3e42;
}

/* 美化暗色主题下拉菜单 */
QComboBox {
    min-height: 24px;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 24px;
    border-left: none;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
}

/* 美化暗色主题标签和按钮 */
QPushButton {
    min-height: 24px;
    font-weight: 500;
}

/* 美化暗色主题对话框和窗口 */
QDialog {
    background-color: #1e1e1e;
    border-radius: 10px;
    padding: 15px;
}

QDialog QLabel {
    color: #e0e0e0;
    font-size: 10pt;
    margin-bottom: 8px;
    font-weight: 500;
}

QDialog QLineEdit {
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 10pt;
    margin-bottom: 12px;
    background-color: #2d2d2d;
    color: #e0e0e0;
    border: 1px solid #3d3d3d;
}

QDialog QListWidget {
    border: 1px solid #3d3d3d;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 15px;
    font-size: 10pt;
    background-color: #252526;
    color: #e0e0e0;
}

QDialog QPushButton {
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 10pt;
}

QDialog QPushButton:default {
    background-color: #007acc;
    color: white;
    border: none;
}

QDialog QPushButton:default:hover {
    background-color: #1c97ea;
}

QDialog QPushButton:default:pressed {
    background-color: #0062a3;
}

QDialog QPushButton:!default {
    background-color: #2d2d30;
    color: #e0e0e0;
    border: 1px solid #3e3e42;
}

QDialog QPushButton:!default:hover {
    background-color: #3e3e42;
    border-color: #007acc;
}

QMainWindow::separator {
    width: 1px;
    height: 1px;
    background-color: #3e3e42;
}

QMainWindow::separator:hover {
    background-color: #007acc;
}

/* 美化暗色主题滑块 */
QSlider::groove:horizontal {
    height: 4px;
    background: #e6e6e6;
    margin: 0px;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #4285f4;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #1967d2;
}

QDialog {
    background-color: #2d2d30;
    border-radius: 10px;
}
QLineEdit {
    padding: 8px 12px;
    border: 1px solid #3e3e42;
    border-radius: 6px;
    background-color: #252526;
    margin-bottom: 10px;
    font-size: 10pt;
    color: #e0e0e0;
}
QLineEdit:focus {
    border-color: #007acc;
    background-color: #333337;
}
QListWidget {
    border: 1px solid #3e3e42;
    border-radius: 8px;
    padding: 5px;
    background-color: #252526;
    color: #e0e0e0;
}
QListWidget::item {
    padding: 8px 12px;
    border-radius: 6px;
    margin: 2px 1px;
}
QListWidget::item:hover {
    background-color: #3e3e42;
}
QListWidget::item:selected {
    background-color: #094771;
    color: #ffffff;
}
QPushButton {
    padding: 8px 20px;
    border-radius: 6px;
    font-size: 8pt;
    font-weight: 500;
    margin: 5px;
    min-width: 90px;
    background-color: #0e639c;
    color: #ffffff;
    border: 1px solid #1177bb;
}
QPushButton#color_button {
    padding: 0px;
    min-width: 28px;
    border-radius: 14px;
    border: 1px solid #3e3e42;
    margin-top: 1px;
}

QWidget {
    color: #f0f0f0;  /* 修改为亮色而不是黑色 */
}
"""

# 明亮主题调色板


def get_light_palette():
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(250, 250, 250))
    palette.setColor(QPalette.WindowText, QColor(44, 62, 80))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(44, 62, 80))
    palette.setColor(QPalette.Text, QColor(44, 62, 80))
    palette.setColor(QPalette.Button, QColor(33, 150, 243))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(33, 150, 243))
    palette.setColor(QPalette.Highlight, QColor(33, 150, 243))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText,
                     QColor(127, 127, 127))
    return palette

# 暗黑主题调色板


def get_dark_palette():
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.AlternateBase, QColor(37, 37, 37))
    palette.setColor(QPalette.ToolTipBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ToolTipText, QColor(224, 224, 224))
    palette.setColor(QPalette.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.Button, QColor(13, 71, 161))
    palette.setColor(QPalette.ButtonText, QColor(224, 224, 224))
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(100, 181, 246))
    palette.setColor(QPalette.Highlight, QColor(100, 181, 246))
    palette.setColor(QPalette.HighlightedText, QColor(45, 45, 45))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText,
                     QColor(128, 128, 128))
    return palette
