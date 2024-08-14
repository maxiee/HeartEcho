import 'package:app/models/corpus.dart';

class ChatSession {
  final List<Message> messages;
  String? path;

  ChatSession(this.messages, {this.path});

  addMessage(Message message) {
    messages.insert(0, message);
  }

  List<Map<String, dynamic>> toHistory() {
    return messages.map((e) => e.toJson()).toList().reversed.toList();
  }
}
