import 'package:flutter/material.dart';

enum Mode {
  Chat,
  LearnKnowledge,
  LearnChat,
}

class GlobalProvider extends ChangeNotifier {
  Mode _mode = Mode.Chat;

  Mode get mode => _mode;

  void changeMode(Mode mode) {
    _mode = mode;
    notifyListeners();
  }
}