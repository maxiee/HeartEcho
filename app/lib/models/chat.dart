class ChatSession {
  final List<ChatMessage> messages;
  String? path;

  ChatSession(this.messages, {this.path});

  addMessage(ChatMessage message) {
    messages.insert(0, message);
  }

  List<Map<String, String>> toHistory() {
    return messages.map((e) => e.toMap()).toList().reversed.toList();
  }
}

class ChatMessage {
  String role;
  String content;

  ChatMessage({required this.role, required this.content});

  Map<String, String> toMap() {
    return {
      'role': role,
      'content': content,
    };
  }
}
