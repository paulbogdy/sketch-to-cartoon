#ifndef CLIENT_CANVASMODEL_H
#define CLIENT_CANVASMODEL_H

#include <QObject>
#include <QImage>
#include <QStack>
#include "../drawing_tools/DrawTool.h"

class CanvasModel: public QObject {
    Q_OBJECT

public:
    explicit CanvasModel(QObject *parent = nullptr);

    QImage image() const;
    QImage shadow() const;
    void setShadow(const QImage &image);
    void setDrawTool(std::shared_ptr<DrawTool> tool);
public slots:
    void setImage(const QImage &image);
    void beginDraw(const QPoint& point);
    void draw(const QPoint& point);
    void endDraw();
    void undo();
    void redo();

signals:
    void imageChanged(const QImage& image);
    void shadowChanged(const QImage& image);
    void undoChanged(bool undo);
    void redoChanged(bool redo);
private:
    QPoint _lastPoint;
    std::shared_ptr<DrawTool> _drawTool;

    QImage _image;
    QImage _shadow;
    QStack<QImage> _undoStack;
    QStack<QImage> _redoStack;
    static const int MAX_UNDO_STACK_SIZE = 25;
};


#endif //CLIENT_CANVASMODEL_H
