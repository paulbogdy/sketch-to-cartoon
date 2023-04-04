#ifndef CLIENT_CANVASMODEL_H
#define CLIENT_CANVASMODEL_H

#include <QObject>
#include <QImage>

class CanvasModel: public QObject {
    Q_OBJECT
public:
    explicit CanvasModel(QObject *parent = nullptr);

    QImage image() const;
    void setImage(const QImage &image);
signals:
    void imageChanged(const QImage& image);
private:
    QImage _image;
};


#endif //CLIENT_CANVASMODEL_H
