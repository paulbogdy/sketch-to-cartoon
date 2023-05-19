//
// Created by maneologu on 03.05.2023.
//

#ifndef CLIENT_FULLSIZEIMAGEWIDGET_H
#define CLIENT_FULLSIZEIMAGEWIDGET_H

#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QCloseEvent>
#include <QDockWidget>
#include <QPushButton>
#include <QFileDialog>


class FullSizeImageWidget : public QDockWidget {
Q_OBJECT

public:
    explicit FullSizeImageWidget(QWidget *parent = nullptr)
            : QDockWidget(parent), _imageLabel(new QLabel(this)), _lastClickedImage(nullptr) {
        QWidget *contentWidget = new QWidget(this);
        QVBoxLayout *layout = new QVBoxLayout(contentWidget);

        _imageLabel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

        QPushButton *exportBtn = new QPushButton(this);
        QIcon exportIcon("resources/icons/export_icon.svg"); // Update with your icon path
        exportBtn->setIcon(exportIcon);
        exportBtn->setToolTip("Export Image");
        exportBtn->setFlat(true);
        exportBtn->setIconSize(QSize(30, 30));
        connect(exportBtn, &QPushButton::clicked, this, &FullSizeImageWidget::exportImage);

        layout->addWidget(_imageLabel);
        layout->addWidget(exportBtn);
        contentWidget->setLayout(layout);

        setWidget(contentWidget);
        setWindowTitle("Full Size Image View");
        hide();
    }

    void showImage(const QImage& image) {
        _imageLabel->setPixmap(QPixmap::fromImage(image));
        show();
    }

    void exportImage() {
        if (_lastClickedImage) {
            QString selectedFilter;
            QString fileName = QFileDialog::getSaveFileName(this, tr("Save Image"), "",
                                                            tr("PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;BMP Image (*.bmp)"),
                                                            &selectedFilter);
            if (!fileName.isEmpty()) {
                QFileInfo info(fileName);
                QString fileExtension = info.suffix().toLower();
                QStringList supportedExtensions = {"png", "jpg", "jpeg", "bmp"};

                if (!supportedExtensions.contains(fileExtension)) {
                    // Add the correct extension based on the selected filter
                    if (selectedFilter.startsWith("PNG")) {
                        fileName += ".png";
                        fileExtension = "png";
                    } else if (selectedFilter.startsWith("JPEG")) {
                        fileName += ".jpg";
                        fileExtension = "jpg";
                    } else if (selectedFilter.startsWith("BMP")) {
                        fileName += ".bmp";
                        fileExtension = "bmp";
                    }
                }

                _lastClickedImage->save(fileName, fileExtension.toUpper().toUtf8().constData());
            }
        }
    }

public slots:
    void imageClicked(const QImage& image) {
        _lastClickedImage = std::make_shared<QImage>(image);
        showImage(image);
    }

private:
    QLabel *_imageLabel;
    std::shared_ptr<QImage> _lastClickedImage;
};



#endif //CLIENT_FULLSIZEIMAGEWIDGET_H
