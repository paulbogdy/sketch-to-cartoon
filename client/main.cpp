#include <QApplication>
#include <QPushButton>
#include <QMainWindow>
#include "UI/mainwindow.h"
#include "settings/Settings.h"

Settings& g_settingsInstance = Settings::getInstance();

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    MainWindow mainWindow;
    mainWindow.show();

    return app.exec();
}
