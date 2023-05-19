//
// Created by maneologu on 02.04.2023.
//

#include "GeneratedImagesView.h"
#include <QPainter>
#include <QDebug>
#include <QMouseEvent>

ImageWidget::ImageWidget(GeneratedImagesModel *generatedImagesModel, QWidget *parent): QWidget(parent)
        , _generatedImagesModel(generatedImagesModel)
        , _imageWidth(128)
        , _imageHeight(128)
        , _spacing(10)
        , _padding(5){
    setMinimumWidth(_imageWidth + 15 + 2 * _padding);
}

void ImageWidget::paintEvent(QPaintEvent *event) {
    QPainter painter(this);
    int y = _spacing;
    for (int i = 0; i < _generatedImagesModel->imageCount(); ++i) {
        QImage img = _generatedImagesModel->imageAt(i).scaled(_imageWidth, _imageHeight, Qt::KeepAspectRatio);
        painter.drawImage(_padding, y, img);  // Fixed x coordinate
        y += _imageHeight + _spacing;
    }
}

void ImageWidget::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    emit sizeChanged();
}

void ImageWidget::adjustSize() {
    int newHeight = _generatedImagesModel->imageCount() * (_imageHeight + _spacing);
    resize(width(), newHeight);
}

void ImageWidget::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        int y = _spacing;
        for (int i = 0; i < _generatedImagesModel->imageCount(); ++i) {
            QRect imageRect(0, y, _imageWidth, _imageHeight);
            if (imageRect.contains(event->pos())) {
                emit imageClicked(_generatedImagesModel->imageAt(i));
                return;
            }
            y += _imageHeight + _spacing;
        }
    }
    QWidget::mousePressEvent(event);
}

GeneratedImagesView::GeneratedImagesView(GeneratedImagesModel *generatedImagesModel, QWidget *parent)
        : QScrollArea(parent)
{
    _imageWidget = new ImageWidget(generatedImagesModel, this);
    setWidget(_imageWidget);

    connect(generatedImagesModel, &GeneratedImagesModel::imagesChanged, this, &GeneratedImagesView::updateImages);
    connect(_imageWidget, &ImageWidget::sizeChanged, this, &GeneratedImagesView::updateGeometrySlot);

    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
}

void GeneratedImagesView::updateGeometrySlot() {
    updateGeometry();
}

void GeneratedImagesView::updateImages() {
    _imageWidget->adjustSize();
    _imageWidget->update();
}

ImageWidget *GeneratedImagesView::imageWidget() const {
    return _imageWidget;
}

