//
// Created by maneologu on 31.03.2023.
//

// You may need to build the project (run Qt uic code generator) to get "ui_MainWindow.h" resolved

#include <QListWidget>
#include <QMenuBar>
#include <QToolBar>
#include <QDockWidget>
#include <QPushButton>
#include "mainwindow.h"
#include "SettingsWindow.h"


MainWindow::MainWindow(QWidget *parent) :
        QMainWindow(parent) {
    // Create the central drawing widget
    QWidget* centralWidget = new QWidget(this);
    QVBoxLayout* centralLayout = new QVBoxLayout(centralWidget);
    centralLayout->setAlignment(Qt::AlignCenter);

    _canvasModel = new CanvasModel();
    _canvasView = new CanvasView(_canvasModel);
    _canvasView->setFixedSize(512, 512);

    _generatedImagesModel = new GeneratedImagesModel();
    _generatedImagesView = new GeneratedImagesView(_generatedImagesModel);

    connect(_generatedImagesModel, &GeneratedImagesModel::imagesChanged, _generatedImagesView, static_cast<void (GeneratedImagesView::*)()>(&GeneratedImagesView::updateImages));

    _controller = new MainController(_canvasModel, _generatedImagesModel);
    QPushButton *generateButton = new QPushButton("Generate", this);
    connect(generateButton, &QPushButton::clicked, this, &MainWindow::onButtonClicked);

    // Add the DrawingWidget to the central layout and set the central widget
    centralLayout->addWidget(_canvasView);
    centralLayout->addWidget(generateButton);
    centralWidget->setLayout(centralLayout);
    setCentralWidget(centralWidget);

    // Create the menu bar
    QMenuBar* menuBar = new QMenuBar(this);
    QMenu* fileMenu = menuBar->addMenu(tr("&File"));

    // Add Settings submenu to the File menu
    QAction* settingsAction = new QAction(tr("&Settings"), this);
    fileMenu->addAction(settingsAction);
    connect(settingsAction, &QAction::triggered, this, &MainWindow::openSettingsWindow);
    // Add more menu items as needed
    setMenuBar(menuBar);

    // Create the left-side toolbar with drawing tools
    QToolBar* toolBar = new QToolBar(this);
    toolBar->setOrientation(Qt::Vertical);
    toolBar->setIconSize(QSize(32, 32));
    // Add tool buttons for pen, crayon, etc.
    addToolBar(Qt::LeftToolBarArea, toolBar);

    // Create the right-side dock widget with a list of sketches
    QDockWidget* sketchDock = new QDockWidget(tr("Sketches"), this);
    QListWidget* sketchList = new QListWidget(sketchDock);
    sketchDock->setWidget(sketchList);
    addDockWidget(Qt::RightDockWidgetArea, sketchDock);


    // Create the bottom dock widget with a carousel of images
    QDockWidget* imageDock = new QDockWidget(tr("Image Carousel"), this);
    imageDock->setWidget(_generatedImagesView);
    addDockWidget(Qt::BottomDockWidgetArea, imageDock);

    // Set the main window properties
    setWindowTitle(tr("Complex Sketching Application"));
    showMaximized();
}

void MainWindow::openSettingsWindow() {
    SettingsWindow settingsWindow(this);
    settingsWindow.exec();
}

void MainWindow::onButtonClicked() {
    _controller->generateImages();
}
