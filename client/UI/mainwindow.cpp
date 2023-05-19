//
// Created by maneologu on 31.03.2023.
//

// You may need to build the project (run Qt uic code generator) to get "ui_MainWindow.h" resolved

#include <QListWidget>
#include <QMenuBar>
#include <QToolBar>
#include <QDockWidget>
#include <QPushButton>
#include <QScrollArea>
#include <QButtonGroup>
#include "mainwindow.h"
#include "SettingsWindow.h"
#include "../views/BottomBar.h"


MainWindow::MainWindow(QWidget *parent) :
        QMainWindow(parent) {

    QString backgroundColor = "rgb(40, 40, 42)";
    QString windowsColor = "rgb(70, 70, 75)";
    QString buttonHoverColor = "rgba(128, 128, 128, 0.2)";
    QString textColor = "rgb(224, 224, 224)";
    QString buttonClickedColor = "#00b4d8";

    // Create the central drawing widget
    QWidget* centralWidget = new QWidget(this);
    QVBoxLayout* centralLayout = new QVBoxLayout(centralWidget);
    centralLayout->setAlignment(Qt::AlignCenter);

    _canvasModel = new CanvasModel(this);
    _canvasView = new CanvasView(_canvasModel, this);
    _canvasView->setFixedSize(512, 512);
    _canvasController = new CanvasController(_canvasModel, _canvasView, this);

    _generatedImagesModel = new GeneratedImagesModel();
    _generatedImagesView = new GeneratedImagesView(_generatedImagesModel);
    _generatedImagesController = new GeneratedImagesController(_canvasModel, _generatedImagesModel, _generatedImagesView, this);

    // Add the DrawingWidget to the central layout and set the central widget
    centralLayout->addWidget(_canvasView);
    centralWidget->setLayout(centralLayout);
    setCentralWidget(centralWidget);

    // Create the menu bar
    QMenuBar* menuBar = new QMenuBar(this);
    QMenu* fileMenu = menuBar->addMenu(tr("&File"));

    // Add Settings submenu to the File menu
    QAction* settingsAction = new QAction(tr("&Settings"), this);
    fileMenu->addAction(settingsAction);
    connect(settingsAction, &QAction::triggered, this, &MainWindow::openSettingsWindow);

    // Add Save Sketch submenu to the File menu
    QAction* saveSketchAction = new QAction(tr("&Save Sketch"), this);
    fileMenu->addAction(saveSketchAction);
    connect(saveSketchAction, &QAction::triggered, _canvasController, &CanvasController::saveSketch);

    // Add Load Sketch submenu to the File menu
    QAction* loadSketchAction = new QAction(tr("&Load Sketch"), this);
    fileMenu->addAction(loadSketchAction);
    connect(loadSketchAction, &QAction::triggered, _canvasController, &CanvasController::loadSketch);

    // Add more menu items as needed
    menuBar->setStyleSheet("QMenuBar {background: " + windowsColor + "; margin-Bottom: 1px; color: " + textColor + "} QMenuBar::item:selected {background: " + buttonHoverColor + "; color: " + textColor + "}");
    fileMenu->setStyleSheet("QMenu {background: " + windowsColor + "; color: " + textColor + "} QMenu::item:selected {background: " + buttonHoverColor + "; color: " + textColor + "}");
    setMenuBar(menuBar);

    // Create the top toolbar for the buttons
    QToolBar* buttonToolBar = new QToolBar(this);
    buttonToolBar->setMovable(false);
    buttonToolBar->setFloatable(false);
    buttonToolBar->setStyleSheet("QToolBar {background: " + windowsColor + ";}"); // Make the toolbar background match the window

    // Create a QWidget and layout to hold the buttons with a stretch
    QWidget* buttonContainer = new QWidget(this);
    QHBoxLayout* buttonLayout = new QHBoxLayout(buttonContainer);
    buttonLayout->setContentsMargins(0, 0, 0, 0); // Remove padding

    // Create the icon buttons
    QPushButton *generateImageBtn = new QPushButton(this);
    QIcon generateImageIcon("resources/icons/generate_images_icon_green.svg"); // Updated icon
    generateImageBtn->setIcon(generateImageIcon);
    generateImageBtn->setToolTip("Generate Image");
    generateImageBtn->setFlat(true);
    generateImageBtn->setIconSize(QSize(30, 30)); // Larger icon size
    generateImageBtn->setStyleSheet("QPushButton {background-color: " + windowsColor + "; border: none; padding: 0px;}" +
                                    "QPushButton:hover {background-color: " + buttonHoverColor + ";}"); // Add hover effect
    connect(generateImageBtn, &QPushButton::clicked, _generatedImagesController, &GeneratedImagesController::generateImages);

    QPushButton *createShadowBtn = new QPushButton(this);
    QIcon createShadowIcon("resources/icons/generate_images_icon_gray_and_green.svg"); // Updated icon
    createShadowBtn->setIcon(createShadowIcon);
    createShadowBtn->setToolTip("Create Shadow");
    createShadowBtn->setFlat(true);
    createShadowBtn->setIconSize(QSize(30, 30)); // Larger icon size
    createShadowBtn->setStyleSheet("QPushButton {background-color: " + windowsColor + "; border: none; padding: 0px;}" +
                                   "QPushButton:hover {background-color: " + buttonHoverColor + ";}"); // Add hover effect
    connect(createShadowBtn, &QPushButton::clicked, _canvasController, &CanvasController::generateShadow);

    QPushButton *undoBtn = new QPushButton(this);
    QIcon undoIcon("resources/icons/undo_gray.svg"); // Updated icon
    undoBtn->setIcon(undoIcon);
    undoBtn->setToolTip("Undo");
    undoBtn->setFlat(true);
    undoBtn->setIconSize(QSize(30, 30)); // Larger icon size
    undoBtn->setStyleSheet("QPushButton {background-color: " + windowsColor + "; border: none; padding: 0px;}"
                           "QPushButton:hover {background-color: " + buttonHoverColor + ";}");
    undoBtn->setEnabled(false);
    connect(undoBtn, &QPushButton::clicked, _canvasController, &CanvasController::undo);
    connect(_canvasModel, &CanvasModel::undoChanged, undoBtn, &QPushButton::setEnabled);

    QPushButton *redoBtn = new QPushButton(this);
    QIcon redoIcon("resources/icons/redo_gray.svg"); // Updated icon
    redoBtn->setIcon(redoIcon);
    redoBtn->setToolTip("Redo");
    redoBtn->setFlat(true);
    redoBtn->setIconSize(QSize(30, 30)); // Larger icon size
    redoBtn->setStyleSheet("QPushButton {background-color: " + windowsColor + "; border: none; padding: 0px;}"
                           "QPushButton:hover {background-color: " + buttonHoverColor + ";}");
    redoBtn->setEnabled(false);
    connect(redoBtn, &QPushButton::clicked, _canvasController, &CanvasController::redo);
    connect(_canvasModel, &CanvasModel::redoChanged, redoBtn, &QPushButton::setEnabled);

    buttonLayout->addStretch();

    // Add the buttons to the layout and set the layout for the container widget
    buttonLayout->addWidget(generateImageBtn);
    buttonLayout->addWidget(createShadowBtn);
    buttonLayout->addWidget(undoBtn);
    buttonLayout->addWidget(redoBtn);
    buttonContainer->setLayout(buttonLayout);

    // Add the button container to the toolbar
    buttonToolBar->addWidget(buttonContainer);

    // Add the top toolbar with buttons to the main window
    addToolBar(Qt::TopToolBarArea, buttonToolBar);

    setStyleSheet("QMainWindow {background: " + backgroundColor + "; color: " + textColor + "}" +
                  "QDockWidget {background: " + windowsColor + " ; color: " + textColor + "; }"); // Make the window background match the toolbar\

    // Add the top toolbar with buttons to the main window
    addToolBar(Qt::TopToolBarArea, buttonToolBar);

    // Create the left toolbar for selecting the brush type
    QToolBar* brushToolBar = new QToolBar(this);
    brushToolBar->setMovable(false);
    brushToolBar->setFloatable(false);
    brushToolBar->setOrientation(Qt::Vertical);
    brushToolBar->setStyleSheet("QToolBar {background: " + windowsColor + ";}"); // Make the toolbar background match the window

    // Create pen button
    QPushButton *penBtn = new QPushButton(this);
    QIcon penIcon("resources/icons/pen_icon.svg");
    penBtn->setIcon(penIcon);
    penBtn->setToolTip("Pen");
    penBtn->setFlat(true);
    penBtn->setIconSize(QSize(30, 30));
    penBtn->setCheckable(true);
    penBtn->setChecked(true);
    penBtn->setStyleSheet("QPushButton {background-color: " + windowsColor + "; border: none; padding: 0px;}" +
                        "QPushButton:hover {background-color: " + buttonHoverColor + ";}" +
                        "QPushButton:checked {background-color: " + buttonClickedColor + ";}");
    connect(penBtn, &QPushButton::clicked, _canvasController, &CanvasController::selectPen);

    // Create eraser button
    QPushButton *eraserBtn = new QPushButton(this);
    QIcon eraserIcon("resources/icons/eraser_icon.svg");
    eraserBtn->setIcon(eraserIcon);
    eraserBtn->setToolTip("Eraser");
    eraserBtn->setFlat(true);
    eraserBtn->setIconSize(QSize(30, 30));
    eraserBtn->setCheckable(true);
    eraserBtn->setStyleSheet("QPushButton {background-color: " + windowsColor + "; border: none; padding: 0px;}" +
                            "QPushButton:hover {background-color: " + buttonHoverColor + ";}" +
                            "QPushButton:checked {background-color: " + buttonClickedColor + ";}");
    connect(eraserBtn, &QPushButton::clicked, _canvasController, &CanvasController::selectEraser);

    QButtonGroup* toolButtonGroup = new QButtonGroup(this);
    toolButtonGroup->setExclusive(true);

    toolButtonGroup->addButton(penBtn);
    toolButtonGroup->addButton(eraserBtn);

    // Add buttons to the brush toolbar
    brushToolBar->addWidget(penBtn);
    brushToolBar->addWidget(eraserBtn);

    // Add the brush toolbar to the left area of the main window
    addToolBar(Qt::LeftToolBarArea, brushToolBar);


    // Create the right dock widget with a carousel of images
    QDockWidget* imageDock = new QDockWidget(tr("Generated Samples"), this);

    imageDock->setWidget(_generatedImagesView);

    // Make it fixed at the right
    imageDock->setAllowedAreas(Qt::RightDockWidgetArea);

    BottomBar* bottomBar = new BottomBar(this, textColor);
    bottomBar->setStyleSheet("QStatusBar { background-color: " + windowsColor + "; border-top: 1px solid; }" +
                             "QStatusBar QLabel { color: " + textColor + "; }");
    this->setStatusBar(bottomBar);

    connect(&MessageBus::getInstance(), &MessageBus::messagePublished, bottomBar, &BottomBar::displayLoadingMessage);

    _fullSizeImageWidget = new FullSizeImageWidget(this);

    // Connect the imageClicked signal of _generatedImagesView to the showImage slot of _fullSizeImageWidget
    connect(_generatedImagesView->imageWidget(), &ImageWidget::imageClicked, _fullSizeImageWidget, &FullSizeImageWidget::imageClicked);

    _fullSizeImageWidget->setAllowedAreas(Qt::RightDockWidgetArea);

    // Add the dock widgets to the main window
    addDockWidget(Qt::RightDockWidgetArea, _fullSizeImageWidget);
    splitDockWidget(_fullSizeImageWidget, imageDock, Qt::Horizontal);

    // Set the main window properties
    setWindowTitle(tr("Complex Sketching Application"));
    showMaximized();
}

void MainWindow::openSettingsWindow() {
    SettingsWindow settingsWindow(this);
    settingsWindow.exec();
}