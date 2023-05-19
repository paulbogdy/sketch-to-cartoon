//
// Created by maneologu on 02.04.2023.
//

#ifndef CLIENT_SETTINGS_H
#define CLIENT_SETTINGS_H

#include <QObject>
#include "../strategies/generative_strategies/GenerativeStrategy.h"
#include <memory>
#include <mutex>

class Settings : public QObject {
Q_OBJECT

public:
    static Settings& getInstance();

    ~Settings() override;
    void setGenerativeStrategy(QString strategyName);
    GenerativeStrategy* getGenerativeStrategy();
    void setImagesToGenerate(int imagesToGenerate);
    int getImagesToGenerate() const;
    void setImagesForShadowDraw(int imagesForShadowDraw);
    int getImagesForShadowDraw() const;
    QString getGenerativeStrategyName() const;

    // Delete copy constructor and assignment operator
    Settings(const Settings&) = delete;
    Settings& operator=(const Settings&) = delete;

private:
    explicit Settings(QObject* parent = nullptr);

    QString _strategyName;
    GenerativeStrategy* _generativeStrategy;
    int _imagesToGenerate;
    int _imagesForShadowDraw;

    static std::shared_ptr<Settings> _instance;
    static std::mutex _mutex;
};


#endif //CLIENT_SETTINGS_H
