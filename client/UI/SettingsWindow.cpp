#include "SettingsWindow.h"
#include "../settings/Settings.h"
#include "../strategies/generative_strategies/GanInversionStrategy.h"
#include "../strategies/generative_strategies/DefaultStrategy.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QComboBox>
#include <QSpinBox>
#include <QLabel>
#include <QPushButton>

SettingsWindow::SettingsWindow(QWidget *parent) : QDialog(parent) {
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // Dropdown for selecting the generative strategy
    QHBoxLayout *strategyLayout = new QHBoxLayout();
    QLabel *strategyLabel = new QLabel(tr("Generative Strategy:"), this);
    QComboBox *strategyComboBox = new QComboBox(this);
    strategyComboBox->setStyleSheet("color: white; background-color: #2d2d2d;");
    strategyComboBox->addItem("Default", QVariant::fromValue<QString>("Default"));
    strategyComboBox->addItem("Gan Inversion", QVariant::fromValue<QString>("Gan Inversion"));
    strategyLayout->addWidget(strategyLabel);
    strategyLayout->addWidget(strategyComboBox);
    mainLayout->addLayout(strategyLayout);
    QString currentStrategy = Settings::getInstance().getGenerativeStrategyName();
    int strategyIndex = strategyComboBox->findData(QVariant::fromValue<QString>(currentStrategy));
    if (strategyIndex != -1) {
        strategyComboBox->setCurrentIndex(strategyIndex);
    }

    // Spinbox for the number of images to generate
    QHBoxLayout *imagesToGenerateLayout = new QHBoxLayout();
    QLabel *imagesToGenerateLabel = new QLabel(tr("Images to Generate:"), this);
    QSpinBox *imagesToGenerateSpinBox = new QSpinBox(this);
    imagesToGenerateSpinBox->setMinimum(1);
    imagesToGenerateLayout->addWidget(imagesToGenerateLabel);
    imagesToGenerateLayout->addWidget(imagesToGenerateSpinBox);
    mainLayout->addLayout(imagesToGenerateLayout);
    int currentImagesToGenerate = Settings::getInstance().getImagesToGenerate();
    imagesToGenerateSpinBox->setValue(currentImagesToGenerate);

    // Spinbox for the number of images to be used in generating the shadow draw image
    QHBoxLayout *imagesForShadowDrawLayout = new QHBoxLayout();
    QLabel *imagesForShadowDrawLabel = new QLabel(tr("Images for Shadow Draw:"), this);
    QSpinBox *imagesForShadowDrawSpinBox = new QSpinBox(this);
    imagesForShadowDrawSpinBox->setMinimum(1);
    imagesForShadowDrawLayout->addWidget(imagesForShadowDrawLabel);
    imagesForShadowDrawLayout->addWidget(imagesForShadowDrawSpinBox);
    mainLayout->addLayout(imagesForShadowDrawLayout);
    int currentImagesForShadowDraw = Settings::getInstance().getImagesForShadowDraw();
    imagesForShadowDrawSpinBox->setValue(currentImagesForShadowDraw);

    // Apply button
    QPushButton *applyButton = new QPushButton(tr("Apply"), this);
    applyButton->setStyleSheet("QPushButton {"
                               "background-color: black;"
                               "color: white;"
                               "}"
                               "QPushButton:hover {"
                               "background-color: #555555;" // Change as needed
                               "}"
                               "QPushButton:pressed {"
                               "background-color: #333333;" // Change as needed
                               "}");
    applyButton->setEnabled(false); // Disabled initially
    mainLayout->addWidget(applyButton);

    // Connect the signals
    connect(strategyComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [=](int index) {
        applyButton->setEnabled(true); // Enable Apply button on change
    });

    connect(imagesToGenerateSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, [=](int value) {
        applyButton->setEnabled(true); // Enable Apply button on change
    });

    connect(imagesForShadowDrawSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, [=](int value) {
        applyButton->setEnabled(true); // Enable Apply button on change
    });

    connect(applyButton, &QPushButton::clicked, this, [=]() {
        // Update the settings
        Settings::getInstance().setGenerativeStrategy(strategyComboBox->itemData(strategyComboBox->currentIndex()).value<QString>());
        Settings::getInstance().setImagesToGenerate(imagesToGenerateSpinBox->value());
        Settings::getInstance().setImagesForShadowDraw(imagesForShadowDrawSpinBox->value());

        // Disable the Apply button until the next change
        applyButton->setEnabled(false);

        accept();
    });

    // Set the layout for the settings window
    setLayout(mainLayout);
}

