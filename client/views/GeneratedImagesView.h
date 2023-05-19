//
// Created by maneologu on 02.04.2023.
//

#ifndef CLIENT_GENERATEDIMAGESVIEW_H
#define CLIENT_GENERATEDIMAGESVIEW_H

#include <QWidget>
#include <QScrollArea>
#include "../models/GeneratedImagesModel.h"


class ImageWidget : public QWidget {
Q_OBJECT
public:
    explicit ImageWidget(GeneratedImagesModel* generatedImagesModel, QWidget* parent = nullptr);

public slots:
    void adjustSize();

signals:
    void sizeChanged();
    void imageClicked(QImage image);

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    GeneratedImagesModel* _generatedImagesModel;
    int _imageWidth;
    int _imageHeight;
    int _spacing;
    int _padding;
};

class GeneratedImagesView: public QScrollArea {
Q_OBJECT
public:
    explicit GeneratedImagesView(GeneratedImagesModel* generatedImagesModel, QWidget* parent = nullptr);
    ImageWidget* imageWidget() const;

public slots:
    void updateImages();
    void updateGeometrySlot();

private:
    ImageWidget* _imageWidget;
};

#endif //CLIENT_GENERATEDIMAGESVIEW_H
