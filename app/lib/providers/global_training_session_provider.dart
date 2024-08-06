import 'package:flutter/foundation.dart';

class GlobalTrainingSessionProvider extends ChangeNotifier {
  bool _isSessionActive = false;
  String? _currentSessionName;

  bool get isSessionActive => _isSessionActive;
  String? get currentSessionName => _currentSessionName;

  void startSession(String sessionName) {
    _isSessionActive = true;
    _currentSessionName = sessionName;
    notifyListeners();
  }

  void endSession() {
    _isSessionActive = false;
    _currentSessionName = null;
    notifyListeners();
  }
}
