//
// Created by maneologu on 19.04.2023.
//

#include "GanInversionStrategy.h"
#include <QBuffer>

GanInversionStrategy::GanInversionStrategy(QObject *parent) : GenerativeStrategy(parent) {

}

void GanInversionStrategy::connectToServer(QLocalSocket &_socket) {
    _socket.connectToServer("/tmp/gan_inversion_server");
    if (!_socket.waitForConnected()) {
        // Handle connection errors
        qDebug() << "Error connecting to Python server:" << _socket.errorString();
    } else {
        // Successfully connected
        qDebug() << "Connected to Python server";
    }
}

void GanInversionStrategy::disconnectFromServer(QLocalSocket &_socket) {
    _socket.disconnectFromServer();
}

void GanInversionStrategy::sendTaskType(QLocalSocket& _socket, QDataStream &stream, TaskType type) {
    qDebug() << "Sending task type... " << type;
    stream << static_cast<quint8>(type);
}

void GanInversionStrategy::sendSketchData(QLocalSocket& _socket, QDataStream &stream, QImage &sketch) {
    // Convert QImage to QByteArray
    QByteArray imageBytes;
    QBuffer imageBuffer(&imageBytes);
    imageBuffer.open(QIODevice::WriteOnly);
    sketch.save(&imageBuffer, "PNG");

    // Send the image size and data
    qDebug() << "Sending image size:" << static_cast<quint32>(imageBytes.size());
    stream << static_cast<quint32>(imageBytes.size());
    _socket.write(imageBytes);
    _socket.waitForBytesWritten();
}

void GanInversionStrategy::sendNumberOfRequestedImages(QLocalSocket& _socket, QDataStream &stream, int numSamples) {
    stream << static_cast<quint32>(numSamples);
}

void GanInversionStrategy::sendShadowComplexity(QLocalSocket& _socket, QDataStream &stream, int numSamples) {
    stream << static_cast<quint32>(numSamples);
}

QImage GanInversionStrategy::retrieveImage(QLocalSocket& _socket) {
    // Wait for the next image size
    qDebug() << "Waiting for image size...";
    _socket.waitForReadyRead();
    QByteArray imageSizeData = _socket.read(4);
    QDataStream imageSizeStream(imageSizeData);
    quint32 imageSize;
    imageSizeStream >> imageSize;
    qDebug() << "Received image size:" << imageSize;

    // Receive the image data
    QByteArray imageData;
    while (imageData.size() < static_cast<int>(imageSize)) {
        qDebug() << "Waiting for image data...";
        _socket.waitForReadyRead();
        qDebug() << "Received image data:" << _socket.bytesAvailable();
        imageData.append(_socket.read(imageSize - imageData.size()));
        qDebug() << "Received image data:" << imageData.size();
    }

    // Convert the received QByteArray to QImage
    qDebug() << "Converting image data to QImage...";
    QImage receivedImage;
    receivedImage.loadFromData(imageData, "PNG");
    qDebug() << "Image converted";
    return receivedImage;
}

QVector<QImage> GanInversionStrategy::generateFromSketch(QImage sketch, int numSamples) {
    QLocalSocket _socket;
    try {
        connectToServer(_socket);
        QDataStream out(&_socket);
        out.setVersion(QDataStream::Qt_5_0);

        sendTaskType(_socket, out, TaskType::GENERATE);
        sendSketchData(_socket, out, sketch);
        sendNumberOfRequestedImages(_socket, out, numSamples);

        QVector<QImage> receivedImages;
        for (quint32 i = 0; i < numSamples; ++i) {
            receivedImages.append(retrieveImage(_socket));
        }

        disconnectFromServer(_socket);
        return receivedImages;
    } catch (std::exception &e) {
        disconnectFromServer(_socket);
        qDebug() << "Error: " << e.what();
        return {};
    }
}

QImage GanInversionStrategy::generateShadow(QImage sketch, int numSamples) {
    QLocalSocket _socket;
    try {
        connectToServer(_socket);
        QDataStream out(&_socket);
        out.setVersion(QDataStream::Qt_5_0);

        sendTaskType(_socket, out, TaskType::SHADOW);
        sendSketchData(_socket, out, sketch);
        sendShadowComplexity(_socket, out, numSamples);

        auto result = retrieveImage(_socket);

        disconnectFromServer(_socket);
        return result;
    } catch (std::exception &e) {
        disconnectFromServer(_socket);
        qDebug() << "Error: " << e.what();
        return {};
    }
}


