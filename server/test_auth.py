import auth


def test_password_hash_is_salted_and_verified():
    first = auth.hash_password("correct horse battery staple")
    second = auth.hash_password("correct horse battery staple")

    assert first != second
    assert auth.verify_password("correct horse battery staple", first)
    assert not auth.verify_password("incorrect password", first)
