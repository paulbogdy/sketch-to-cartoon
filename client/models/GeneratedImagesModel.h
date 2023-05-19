//
// Created by maneologu on 02.04.2023.
//

#ifndef CLIENT_GENERATEDIMAGESMODEL_H
#define CLIENT_GENERATEDIMAGESMODEL_H

#include <QObject>
#include <QImage>

class GeneratedImagesModel: public QObject {
    Q_OBJECT

    void setShadow(QImage image);

public:
    explicit GeneratedImagesModel(QObject* parent = nullptr);

    int imageCount() const;
    QImage imageAt(int index) const;
    void addImage(const QImage& image);
    void setImages(const QVector<QImage>& images);
    void removeImage(int index);

signals:
    void imagesChanged();

private:
    QVector<QImage> _images;
};


#endif //CLIENT_GENERATEDIMAGESMODEL_H
