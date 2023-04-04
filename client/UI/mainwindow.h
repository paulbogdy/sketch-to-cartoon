//
// Created by maneologu on 31.03.2023.
//

#ifndef CLIENT_MAINWINDOW_H
#define CLIENT_MAINWINDOW_H

#include <QMainWindow>
#include <QVBoxLayout>
#include "../views/CanvasView.h"
#include "../views/GeneratedImagesView.h"
#include "../controllers/MainController.h"


QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
public slots:
    void onButtonClicked();

private:
    CanvasView* _canvasView;
    CanvasModel* _canvasModel;
    GeneratedImagesView* _generatedImagesView;
    GeneratedImagesModel* _generatedImagesModel;
    MainController* _controller;

    void openSettingsWindow();
};


#endif //CLIENT_MAINWINDOW_H
