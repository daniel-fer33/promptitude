import pytest
from promptitude import guidance


def test_exists():
    """Test the behavior of 'exists'."""
    program = guidance("""{{#if (exists 'variable')}}Variable exists{{else}}Variable does not exist{{/if}}""")

    # Test when the variable exists
    output = program(variable='some value')
    assert str(output) == "Variable exists"

    # Test when the variable does not exist
    output = program()
    assert str(output) == "Variable does not exist"


def test_exists_nested_keys():
    """Test 'exists' with nested keys."""

    # Test when nested variable exists
    program = guidance("""{{#if (exists 'user.name')}}Name exists{{else}}Name does not exist{{/if}}""")
    output = program(user={'name': 'Alice'})
    assert str(output) == "Name exists"

    # Test when intermediate key exists but final key does not
    program = guidance("""{{#if (exists 'user.name')}}Name exists{{else}}Name does not exist{{/if}}""")
    output = program(user={})
    assert str(output) == "Name does not exist"

    # Test when top-level key does not exist
    program = guidance("""{{#if (exists 'user.name')}}Name exists{{else}}Name does not exist{{/if}}""")
    output = program()
    assert str(output) == "Name does not exist"


def test_exists_non_dict_intermediate():
    """Test 'exists' when intermediate key is not a dict."""
    program = guidance("""{{#if (exists 'user.name')}}Name exists{{else}}Name does not exist{{/if}}""")

    # 'user' is not a dict, so accessing 'user.name' should fail gracefully
    output = program(user='not a dict')
    assert str(output) == "Name does not exist"


def test_exists_check_nested_true():
    """Test 'exists' with check_nested set to True."""
    program = guidance(
        """{{#if (exists 'user.name' check_nested=True)}}Name exists{{else}}Name does not exist{{/if}}""")

    # Test when 'user' is not a dict
    output = program(user='not a dict')
    assert isinstance(output._exception, ValueError)

    # Test when 'user' is missing entirely
    output = program()
    assert isinstance(output._exception, ValueError)

    # Test when 'user.name' is missing
    output = program(user={})
    assert str(output) == "Name does not exist"


def test_exists_missing_intermediate_key():
    """Test 'exists' when an intermediate key is missing."""
    program = guidance(
        """{{#if (exists 'user.name.first')}}First name exists{{else}}First name does not exist{{/if}}""")

    # 'user.name' is missing, so accessing 'user.name.first' should fail gracefully
    output = program(user={})
    assert str(output) == "First name does not exist"

    # 'user.name' exists but is not a dict
    output = program(user={'name': 'Alice'})
    assert str(output) == "First name does not exist"

    # 'user.name.first' exists
    output = program(user={'name': {'first': 'Alice'}})
    assert str(output) == "First name exists"