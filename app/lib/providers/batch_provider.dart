import 'package:app/api/api.dart';
import 'package:app/models/chat.dart';
import 'package:flutter/widgets.dart';

/// 攒够一定数量的语料库，就可以进行训练
class BatchProvider extends ChangeNotifier {
  List<dynamic> corpus = []; // 语料库

  void addChatSession(ChatSession chatSession) {
    corpus.insert(0, chatSession);
    notifyListeners();
  }

  void addKnowledge(String knowledge) {
    corpus.insert(0, knowledge);
    notifyListeners();
  }

  void train() {
    API.learn(corpus.whereType<ChatSession>().toList(), corpus.whereType<String>().toList());
  }

  void clear() {
    corpus.clear();
    notifyListeners();
  }
}