//
// Created by maneologu on 02.04.2023.
//

#include "GeneratedImagesView.h"
#include <QPainter>
#include <QMouseEvent>

GeneratedImagesView::GeneratedImagesView(GeneratedImagesModel *generatedImagesModel, QWidget *parent)
    : QWidget(parent)
    , _generatedImagesModel(generatedImagesModel)
    , _offset(0)
    , _imageWidth(128)
    , _imageHeight(128)
    , _spacing(10)
    {
    connect(_generatedImagesModel, &GeneratedImagesModel::imagesChanged, this, &GeneratedImagesView::updateImages);
}

QSize GeneratedImagesView::sizeHint() const {
    return QSize(_imageWidth * 5 + _spacing * 4, _imageHeight);
}

void GeneratedImagesView::updateImages() {

}

void GeneratedImagesView::paintEvent(QPaintEvent *event) {
    QPainter painter(this);
    int x = _spacing - _offset;
    for (int i = 0; i < _generatedImagesModel->imageCount(); ++i) {
        QImage img = _generatedImagesModel->imageAt(i).scaled(_imageWidth, _imageHeight, Qt::KeepAspectRatio);
        painter.drawImage(x, (height() - img.height()) / 2, img);
        x += _imageWidth + _spacing;
    }
}


