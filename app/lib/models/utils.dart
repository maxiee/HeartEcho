String idFromJson(dynamic id) {
  if (id is Map<String, dynamic> && id.containsKey('\$oid')) {
    return id['\$oid'];
  } else if (id is String) {
    return id;
  }
  throw const FormatException('Invalid MongoDB id format');
}

DateTime dateTimeFromJson(dynamic date) {
  if (date is String) {
    return DateTime.parse(date);
  } else if (date is Map<String, dynamic> && date.containsKey('\$date')) {
    if (date['\$date'] is String) {
      return DateTime.parse(date['\$date']);
    } else if (date['\$date'] is int) {
      return DateTime.fromMillisecondsSinceEpoch(date['\$date']);
    }
  }
  throw const FormatException('Invalid date format');
}
