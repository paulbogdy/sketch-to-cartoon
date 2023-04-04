//
// Created by maneologu on 02.04.2023.
//

#ifndef CLIENT_GENERATEDIMAGESVIEW_H
#define CLIENT_GENERATEDIMAGESVIEW_H

#include <QWidget>
#include "../models/GeneratedImagesModel.h"

class GeneratedImagesView: public QWidget {
    Q_OBJECT
public:
    explicit GeneratedImagesView(GeneratedImagesModel* generatedImagesModel, QWidget* parent = nullptr);
    QSize sizeHint() const override;

public slots:
    void updateImages();

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    GeneratedImagesModel* _generatedImagesModel;
    int _offset;
    int _imageWidth;
    int _imageHeight;
    int _spacing;
};


#endif //CLIENT_GENERATEDIMAGESVIEW_H
